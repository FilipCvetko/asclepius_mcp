"""Medical contacts module - Notion-backed, cached."""

import os
from typing import Any, Dict, List, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cache import is_cache_fresh, read_cache, write_cache

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
    # Try fresh cache
    if is_cache_fresh("contacts"):
        cached = read_cache("contacts")
        if cached:
            return cached
    
    # Try fetch from Notion
    try:
        contacts = fetch_contacts_from_notion()
        if contacts:
            write_cache("contacts", contacts)
            return contacts
    except Exception as e:
        print(f"Warning: Failed to fetch contacts from Notion: {e}")
    
    # Fall back to stale cache
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


def _get_phone_number(query: str) -> Dict[str, Any]:
    """Get emergency medical contact information and phone numbers instantly.
    
    This is your go-to tool for finding Slovenian medical contacts in emergency situations.
    Whether you need a hospital phone number, specialist contact, or emergency service,
    this tool searches through a comprehensive database of medical facilities and returns
    ALL relevant contact information with phone numbers that match your query well.
    
    USE THIS TOOL WHEN:
    - Someone needs immediate medical help and you need hospital contacts
    - Looking for specialist phone numbers (cardiologist, pediatrician, etc.)
    - Finding emergency services (ambulance, poison control, urgent care)
    - Locating clinics or medical departments in specific cities
    - Getting contact info for any medical facility in Slovenia
    
    EXAMPLES - Try these queries:
    • "heart attack emergency" → Finds cardiology emergency contacts
    • "child fever" → Pediatric emergency numbers
    • "UKC Ljubljana phone" → University Medical Center contact
    • "poison control Slovenia" → Poison control center
    • "dental emergency Maribor" → Emergency dental care
    • "ambulance Ljubljana" → Ambulance services
    • "eye clinic" → Ophthalmology departments
    • "Maribor hospital" → Hospital contacts in Maribor
    
    The tool understands medical terms, hospital names, and emergency situations.
    It will return ALL relevant matches with similarity scores above 0.33.
    """
    if not CONTACTS or VECTORIZER is None or VECTORS is None:
        return {"found": False, "error": "No contacts available"}
    
    try:
        query_vector = VECTORIZER.transform([query])
        similarities = cosine_similarity(query_vector, VECTORS).flatten()
        
        # Get all contacts with similarity > 0.33
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
        
        # Sort by confidence (highest first)
        matching_contacts.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not matching_contacts:
            return {"found": False, "error": "No suitable matches found", "query": query}
        
        return {
            "found": True,
            "count": len(matching_contacts),
            "contacts": matching_contacts,
            "_citation_instruction": (
                "IMPORTANT: For each contact, show the name, phone number, and notes. "
                "Show the confidence score. Present ALL matching contacts."
            ),
        }
    except Exception as e:
        return {"found": False, "error": str(e)[:100]}
