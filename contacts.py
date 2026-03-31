"""Medical contacts module - Notion-backed, cached."""

import os
from typing import Any, Dict, List, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cache import is_cache_fresh, read_cache, write_cache
from fastmcp.tools.tool import ToolResult

# Configuration
NOTION_COLS = {
    "name": "Kdo?",
    "phone": "Telefonska številka",
    "notes": "Opomba",
}

HEADERS = {
    "Authorization": f"Bearer {os.getenv('NOTION_API_KEY', '')}",
    "Notion-Version": "2026-03-11",
    "Content-Type": "application/json",
}

# Module-level state
CONTACTS: List[Dict[str, Any]] = []
VECTORIZER: Optional[TfidfVectorizer] = None
VECTORS: Optional[Any] = None


def get_notion_data_source_id() -> str:
    """Fetch database and extract data_source_id from it."""
    database_id = os.getenv("NOTION_DATABASE_ID")
    if not database_id:
        raise ValueError("NOTION_DATABASE_ID env var required")

    response = requests.get(
        f"https://api.notion.com/v1/databases/{database_id}",
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()
    db = response.json()

    data_sources = db.get("data_sources", [])
    if not data_sources:
        raise ValueError("No data sources found in database")

    return data_sources[0]["id"]


def extract_text(prop: Any) -> str:
    """Extract text from Notion property value."""
    if not prop:
        return ""
    if isinstance(prop, str):
        return prop

    prop_type = prop.get("type", "")
    if prop_type == "title":
        return "".join([t.get("plain_text", "") for t in prop.get("title", [])])
    elif prop_type == "rich_text":
        return "".join([t.get("plain_text", "") for t in prop.get("rich_text", [])])
    elif prop_type == "phone_number":
        return prop.get("phone_number", "")

    return ""


def fetch_contacts_from_notion() -> List[Dict[str, Any]]:
    """Query Notion data_source and fetch all contacts with pagination."""
    data_source_id = get_notion_data_source_id()
    contacts = []
    cursor = None

    while True:
        body = {"start_cursor": cursor} if cursor else {}
        response = requests.post(
            f"https://api.notion.com/v1/data_sources/{data_source_id}/query",
            headers=HEADERS,
            json=body,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        for page in data.get("results", []):
            props = page.get("properties", {})
            name = extract_text(props.get(NOTION_COLS["name"]))
            phone = extract_text(props.get(NOTION_COLS["phone"]))
            notes = extract_text(props.get(NOTION_COLS["notes"]))

            if name or phone:
                contacts.append({
                    "name": name.encode("utf-8", errors="ignore").decode("utf-8"),
                    "phone": phone.encode("utf-8", errors="ignore").decode("utf-8"),
                    "notes": notes.encode("utf-8", errors="ignore").decode("utf-8"),
                })

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return contacts


def load_contacts() -> List[Dict[str, Any]]:
    """Load contacts: fresh cache → fetch Notion → stale cache → empty."""
    if is_cache_fresh("contacts"):
        cached = read_cache("contacts")
        if cached:
            return cached

    try:
        contacts = fetch_contacts_from_notion()
        if contacts:
            write_cache("contacts", contacts)
            return contacts
    except Exception as e:
        print(f"Warning: Failed to fetch contacts from Notion: {e}")

    cached = read_cache("contacts")
    return cached or []


def initialize_contacts() -> None:
    """Initialize contacts and vectorizer on startup."""
    global CONTACTS, VECTORIZER, VECTORS

    print("Loading contacts...")
    CONTACTS = load_contacts()
    print(f"Loaded {len(CONTACTS)} contacts")

    if CONTACTS:
        corpus = [f"{c.get('name', '')} {c.get('phone', '')} {c.get('notes', '')}" for c in CONTACTS]
        VECTORIZER = TfidfVectorizer()
        VECTORIZER.fit(corpus)
        VECTORS = VECTORIZER.transform(corpus)
        print("Contacts vectorizer initialized successfully")
    else:
        print("No contacts loaded, vectorizer not initialized")


# ── Formatting helpers ──────────────────────────────────────────────

def _format_contacts_md(data: dict) -> str:
    if not data.get("found"):
        return f"No contacts found. {data.get('error', '')}"

    lines = [f"**Kontakti** | {data['count']} results", ""]

    lines.append("| Ime | Telefon | Opomba | Zaupanje |")
    lines.append("|-----|---------|--------|----------|")

    for c in data["contacts"]:
        name = c.get("name", "")
        phone = c.get("phone", "")
        notes = c.get("notes", "")
        conf = round(c.get("confidence", 0), 2)
        lines.append(f"| {name} | {phone} | {notes} | {conf} |")

    return "\n".join(lines)


# ── Tool functions ──────────────────────────────────────────────────

def _get_phone_number(query: str) -> ToolResult:
    """Get emergency medical contact information and phone numbers instantly."""
    if not CONTACTS or VECTORIZER is None or VECTORS is None:
        err = {"found": False, "error": "No contacts available"}
        return ToolResult(content=_format_contacts_md(err), structured_content=err)

    try:
        query_vector = VECTORIZER.transform([query])
        similarities = cosine_similarity(query_vector, VECTORS).flatten()

        matching_contacts = []
        for idx, score in enumerate(similarities):
            if score > 0.33:
                contact = CONTACTS[idx]
                matching_contacts.append({
                    "name": contact["name"],
                    "phone": contact["phone"],
                    "notes": contact["notes"],
                    "confidence": float(score),
                })

        matching_contacts.sort(key=lambda x: x["confidence"], reverse=True)

        if not matching_contacts:
            err = {"found": False, "error": "No suitable matches found", "query": query}
            return ToolResult(content=_format_contacts_md(err), structured_content=err)

        data = {
            "found": True,
            "count": len(matching_contacts),
            "contacts": matching_contacts,
        }
        return ToolResult(content=_format_contacts_md(data), structured_content=data)
    except Exception as e:
        err = {"found": False, "error": str(e)[:100]}
        return ToolResult(content=_format_contacts_md(err), structured_content=err)
