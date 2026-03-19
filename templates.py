"""Clinical note formatting templates module - Notion-backed, cached."""

import os
from typing import Any, Dict, List, Optional

import requests

from cache import is_cache_fresh, read_cache, write_cache

# Configuration
CACHE_NAME = "templates"
CACHE_TTL_HOURS = 24

NOTION_COLS = {
    "name": "Ime predloge",
    "sections": "Sekcije",
    "instructions": "Navodila",
    "example": "Primer",
}

# Module-level state
TEMPLATES: List[Dict[str, Any]] = []


def _get_notion_headers() -> Dict[str, str]:
    """Get Notion API headers."""
    return {
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY', '')}",
        "Notion-Version": "2026-03-11",
        "Content-Type": "application/json",
    }


def _extract_text(prop: Any) -> str:
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

    return ""


def _get_data_source_id() -> str:
    """Fetch templates database and extract data_source_id."""
    database_id = os.getenv("NOTION_TEMPLATES_DATABASE_ID")
    if not database_id:
        raise ValueError("NOTION_TEMPLATES_DATABASE_ID env var required")

    headers = _get_notion_headers()
    response = requests.get(
        f"https://api.notion.com/v1/databases/{database_id}",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    db = response.json()

    data_sources = db.get("data_sources", [])
    if not data_sources:
        raise ValueError("No data sources found in templates database")

    return data_sources[0]["id"]


def _fetch_templates_from_notion() -> List[Dict[str, Any]]:
    """Query templates database with pagination."""
    data_source_id = _get_data_source_id()
    headers = _get_notion_headers()
    templates = []
    cursor = None

    while True:
        body = {"start_cursor": cursor} if cursor else {}
        response = requests.post(
            f"https://api.notion.com/v1/data_sources/{data_source_id}/query",
            headers=headers,
            json=body,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        for page in data.get("results", []):
            props = page.get("properties", {})
            name = _extract_text(props.get(NOTION_COLS["name"]))
            sections = _extract_text(props.get(NOTION_COLS["sections"]))
            instructions = _extract_text(props.get(NOTION_COLS["instructions"]))
            example = _extract_text(props.get(NOTION_COLS["example"]))

            if name:
                templates.append({
                    "name": name.strip(),
                    "sections": sections.strip(),
                    "instructions": instructions.strip(),
                    "example": example.strip(),
                })

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return templates


def load_templates() -> List[Dict[str, Any]]:
    """Load templates: fresh cache -> fetch Notion -> stale cache -> empty."""
    if is_cache_fresh(CACHE_NAME, max_age_hours=CACHE_TTL_HOURS):
        cached = read_cache(CACHE_NAME)
        if cached:
            return cached

    # Try fetch from Notion
    try:
        templates = _fetch_templates_from_notion()
        if templates:
            write_cache(CACHE_NAME, templates)
            return templates
    except Exception as e:
        print(f"Warning: Failed to fetch templates from Notion: {e}")

    # Fall back to stale cache
    cached = read_cache(CACHE_NAME)
    return cached or []


def initialize_templates() -> None:
    """Initialize templates module."""
    global TEMPLATES

    db_id = os.getenv("NOTION_TEMPLATES_DATABASE_ID")
    if not db_id:
        print("Templates module skipped (NOTION_TEMPLATES_DATABASE_ID not set)")
        return

    print("Loading clinical note templates...")
    TEMPLATES = load_templates()
    print(f"Loaded {len(TEMPLATES)} templates")


def _get_template(name: str) -> Dict[str, Any]:
    """Fuzzy match template by name."""
    if not TEMPLATES:
        # Lazy refresh
        initialize_templates()
        if not TEMPLATES:
            return {"found": False, "error": "No templates available"}

    name = name.strip().lower()
    if not name:
        return {"found": False, "error": "Template name must not be empty"}

    # Exact match first
    for t in TEMPLATES:
        if t["name"].lower() == name:
            return {"found": True, "template": t}

    # Substring match
    matches = [t for t in TEMPLATES if name in t["name"].lower()]
    if len(matches) == 1:
        return {"found": True, "template": matches[0]}
    elif len(matches) > 1:
        return {
            "found": True,
            "count": len(matches),
            "templates": matches,
        }

    # Partial word match
    matches = [t for t in TEMPLATES if any(name in word.lower() for word in t["name"].split())]
    if matches:
        return {
            "found": True,
            "count": len(matches),
            "templates": matches,
        }

    return {"found": False, "error": f"No template matching '{name}'", "available": [t["name"] for t in TEMPLATES]}


def _list_templates() -> str:
    """List all available templates."""
    if not TEMPLATES:
        initialize_templates()
        if not TEMPLATES:
            return "No templates available. Set NOTION_TEMPLATES_DATABASE_ID env var."

    lines = []
    for t in TEMPLATES:
        sections = t.get("sections", "")
        lines.append(f"- **{t['name']}**: {sections}")
    return "\n".join(lines)


def _format_note_prompt(template_name: str, raw_text: str) -> str:
    """Build structured prompt for formatting clinical notes."""
    result = _get_template(template_name)

    if not result.get("found"):
        available = ", ".join(t["name"] for t in TEMPLATES) if TEMPLATES else "none"
        return f"Template '{template_name}' not found. Available templates: {available}"

    template = result.get("template") or result.get("templates", [{}])[0]

    sections = template.get("sections", "")
    instructions = template.get("instructions", "")
    example = template.get("example", "")

    prompt_parts = [
        f"Format the following dictated clinical note using the template '{template['name']}'.",
        f"\n## Sections\n{sections}" if sections else "",
        f"\n## Formatting Instructions\n{instructions}" if instructions else "",
        f"\n## Example\n{example}" if example else "",
        f"\n## Raw Dictated Text\n{raw_text}",
        "\nPlease format the above text into the specified sections. "
        "Maintain all clinical details. Use professional medical language. "
        "If information for a section is not available, note 'Ni podatkov'.",
    ]

    return "\n".join(p for p in prompt_parts if p)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    print("Testing templates module...")
    initialize_templates()

    print("\n--- List templates ---")
    print(_list_templates())
