# FastMCP Server - Medical Contacts (Notion-backed, cached)
# Pull contacts from Notion, cache locally, search via cosine similarity

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import numpy as np
from dotenv import load_dotenv
from fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
mcp = FastMCP("medical-tools")

# Configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "contacts.json"
CACHE_MAX_AGE = 7 * 24 * 60 * 60  # 1 week

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


def is_cache_fresh() -> bool:
    """Check if cache exists and is less than 1 week old."""
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age <= CACHE_MAX_AGE


def read_cache() -> Optional[List[Dict[str, Any]]]:
    """Read contacts from cache file."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def write_cache(contacts: List[Dict[str, Any]]) -> None:
    """Write contacts to cache file."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(contacts, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_contacts() -> List[Dict[str, Any]]:
    """Load contacts: fresh cache → fetch Notion → stale cache → empty."""
    # Try fresh cache
    if is_cache_fresh():
        cached = read_cache()
        if cached:
            return cached
    
    # Try fetch from Notion
    try:
        contacts = fetch_contacts_from_notion()
        if contacts:
            write_cache(contacts)
            return contacts
    except Exception as e:
        print(f"Warning: Failed to fetch from Notion: {e}")
    
    # Fall back to stale cache
    cached = read_cache()
    return cached or []


# Load contacts on startup
print("Loading contacts on startup...")
CONTACTS = load_contacts()
print(f"Loaded {len(CONTACTS)} contacts")
VECTORIZER = None
VECTORS = None

if CONTACTS:
    corpus = [f"{c.get('name', '')} {c.get('phone', '')} {c.get('notes', '')}" for c in CONTACTS]
    VECTORIZER = TfidfVectorizer()
    VECTORIZER.fit(corpus)
    VECTORS = VECTORIZER.transform(corpus)
    print("Vectorizer initialized successfully")
else:
    print("No contacts loaded, vectorizer not initialized")

@mcp.tool()
def get_phone_number(query: str) -> Dict[str, Any]:
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
        }
    except Exception as e:
        return {"found": False, "error": str(e)[:100]}

if __name__ == "__main__":
    mcp.run()