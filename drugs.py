"""Medical drugs module - CBZ API (Centralna baza zdravil)."""

import base64
import hashlib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from cache import is_cache_fresh, read_cache, write_cache

# Configuration
CBZ_API_BASE = "https://www.cbz.si/cbz/bazazdr2.nsf/Search"
DRUG_PAGE_CACHE_TTL = 24  # hours


def search_drugs_cbz(query: str, max_results: int = 301) -> List[Dict[str, Any]]:
    """Search for drugs in the CBZ (Centralna baza zdravil) database.
    
    Args:
        query: Drug name or search term
        max_results: Maximum number of results (default 301)
    
    Returns:
        List of drug results from CBZ filtered by cosine similarity (>= 0.6)
    """
    try:
        # Build the search query - search in drug name field
        # Format: ([TXIMELAS1]=_DRUGNAME*)
        search_query = f"([TXIMELAS1]=_{quote(query, safe='')}*)"
        
        # Build full URL string
        url = f"{CBZ_API_BASE}?SearchView&Query={search_query}&SearchOrder=4&SearchMax={max_results}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        smpc_map = _extract_smpc_urls(response.text)
        drugs = _parse_cbz_html(response.text, query)
        # Attach SmPC URLs to matching drugs
        for drug in drugs:
            drug["smpc_url"] = smpc_map.get(drug.get("href", ""))
        return drugs
    except Exception as e:
        print(f"Warning: Failed to fetch drugs from CBZ: {e}")
        return []


def _parse_cbz_html(html: str, query: str) -> List[Dict[str, Any]]:
    """Parse HTML response from CBZ API and extract drug information."""
    drugs = []
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Find all TD elements with class 'textbarva0'
        tds = soup.find_all("td", class_="textbarva0")
        
        if not tds:
            return drugs
        
        # Extract drug names and links
        for td in tds:
            try:
                # Find all anchor tags within the td
                links = td.find_all("a")
                
                if len(links) < 2:
                    continue
                
                # The second link contains the drug name and href to the document
                drug_link = links[1]
                href = drug_link.get("href", "").strip()
                drug_name = drug_link.get_text(strip=True)
                
                # Skip if no name or href
                if not drug_name or not href:
                    continue
                
                # Clean up the href (make it absolute if needed)
                if href.startswith("/cbz"):
                    href = f"https://www.cbz.si{href}"
                
                # Fetch and parse the drug detail page
                drug_details = _fetch_and_parse_drug_page(href)
                
                drug_data = {
                    "name": drug_name.encode("utf-8", errors="ignore").decode("utf-8"),
                    "href": href,
                    "title": drug_link.get("title", "").encode("utf-8", errors="ignore").decode("utf-8"),
                }
                
                # Merge in the parsed details
                if drug_details:
                    drug_data.update(drug_details)
                
                drugs.append(drug_data)
            except Exception as e:
                continue
        
    except Exception as e:
        print(f"Error parsing CBZ HTML: {e}")
    
    return drugs


def _fetch_and_parse_drug_page(href: str) -> Optional[Dict[str, Any]]:
    """Fetch and parse individual drug detail page, returning cleaned content."""
    try:
        response = requests.get(href, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Find the main content area - typically in body or a main container
        # Remove navigation, header, footer if possible
        for element in soup.find_all(["nav", "header", "footer"]):
            element.decompose()
        
        # Get the main body content or largest content div
        main_content = soup.find("body")
        if not main_content:
            main_content = soup
        
        # Get text with preserved structure
        text = main_content.get_text(separator="\n", strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines)
        
        # Limit to reasonable size for LLM consumption
        if len(content) > 3000:
            content = content[:3000] + "\n... [content truncated]"
        
        return {
            "content": content.encode("utf-8", errors="ignore").decode("utf-8"),
            "source": href
        }
        
    except Exception as e:
        print(f"Error fetching drug page {href}: {e}")
        return None


def initialize_drugs() -> None:
    """Initialize drugs module (no-op for direct API calls)."""
    print("Drugs module initialized (direct CBZ API mode)")


def _get_drug_info(query: str) -> Dict[str, Any]:
    """Get drug information from CBZ database.
    
    Search for drugs by name in the Slovenian Central Drug Database (CBZ).
    Returns all matching drugs from the search results with their clickable links.
    
    USE THIS TOOL WHEN:
    - Looking up drug information by name
    - Finding available forms (tablets, injections, drops, etc.)
    - Getting links to full drug documentation
    
    EXAMPLES - Try these queries:
    • "Plivit D3" → Vitamin D preparations
    • "Aspirin" → Pain/fever medication
    • "Paracetamol" → Acetaminophen variants
    • "Amoksicilina" → Antibiotic
    • "Ibuprofen" → Anti-inflammatory
    """
    if not query or len(query.strip()) < 2:
        return {"found": False, "error": "Query must be at least 2 characters"}
    
    try:
        results = search_drugs_cbz(query.strip())
        
        if not results:
            return {
                "found": False,
                "error": "No drugs found for your query",
                "query": query
            }
        
        return {
            "found": True,
            "count": len(results),
            "query": query,
            "drugs": results,
        }
    except Exception as e:
        return {"found": False, "error": str(e)[:100]}


def _search_drug_urls(query: str, max_results: int = 301) -> List[tuple]:
    """Search CBZ and return (name, url) pairs without fetching each drug page.

    Returns:
        List of (drug_name, drug_url) tuples
    """
    try:
        search_query = f"([TXIMELAS1]=_{quote(query, safe='')}*)"
        url = f"{CBZ_API_BASE}?SearchView&Query={search_query}&SearchOrder=4&SearchMax={max_results}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        tds = soup.find_all("td", class_="textbarva0")

        results = []
        for td in tds:
            links = td.find_all("a")
            if len(links) < 2:
                continue
            drug_link = links[1]
            href = drug_link.get("href", "").strip()
            name = drug_link.get_text(strip=True)
            if not name or not href:
                continue
            if href.startswith("/cbz"):
                href = f"https://www.cbz.si{href}"
            results.append((
                name.encode("utf-8", errors="ignore").decode("utf-8"),
                href
            ))
        return results
    except Exception as e:
        print(f"Warning: Failed to search CBZ: {e}")
        return []


SMPC_CLINICAL_SECTIONS = {
    "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.8", "4.9", "5.1", "5.2",
}


def _extract_smpc_urls(html: str) -> Dict[str, str]:
    """Extract SmPC PDF URLs from CBZ search results HTML.

    Returns a dict mapping drug page URLs to absolute SmPC PDF URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    smpc_map: Dict[str, str] = {}

    for btn in soup.find_all("input", class_="button_smpc"):
        onclick = btn.get("onclick", "")
        m = re.search(r"window\.location='([^']+)'", onclick)
        if not m:
            continue
        smpc_path = m.group(1)
        smpc_url = f"https://www.cbz.si{smpc_path}" if smpc_path.startswith("/") else smpc_path

        # Walk up to find the enclosing element that also contains the drug link
        parent = btn
        drug_href = None
        for _ in range(10):
            parent = parent.parent
            if parent is None:
                break
            link = parent.find("a", href=re.compile(r"opendocument"))
            if link:
                drug_href = link.get("href", "")
                break

        if drug_href:
            drug_href = drug_href.strip()
            if drug_href.startswith("/"):
                drug_href = f"https://www.cbz.si{drug_href}"
            smpc_map[drug_href] = smpc_url

    return smpc_map


def _fetch_smpc_pdf(url: str) -> Optional[bytes]:
    """Download SmPC PDF bytes, using cache when available."""
    cache_key = f"smpc_pdf_{hashlib.md5(url.encode()).hexdigest()}"

    if is_cache_fresh(cache_key, max_age_hours=DRUG_PAGE_CACHE_TTL):
        cached = read_cache(cache_key)
        if cached and isinstance(cached, list) and cached:
            try:
                return base64.b64decode(cached[0])
            except Exception:
                pass

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        pdf_bytes = response.content
        write_cache(cache_key, [base64.b64encode(pdf_bytes).decode("ascii")])
        return pdf_bytes
    except Exception as e:
        print(f"Error fetching SmPC PDF {url}: {e}")
        return None


def _parse_smpc_sections(pdf_bytes: bytes) -> Dict[str, Dict[str, str]]:
    """Extract text from SmPC PDF and split into numbered sections.

    Handles two common SmPC heading formats:
    - Same line:  ``\\n4.1 Terapevtske indikacije\\n``
    - Split line: ``\\n4.1 \\n Terapevtske indikacije\\n``

    Filters out inline references like "poglavje 4.4" that aren't real headings.

    Returns dict mapping section number to {"title": ..., "text": ...}.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Match section headings — number at line start, title on same or next line
    # Pattern: \n{num}\s*\n\s*{title}\n  OR  \n{num} {title}\n
    header_pattern = re.compile(
        r"\n(\d+\.[\d.]*)\s*\n\s*([A-ZČŠŽĆĐ][^\n]+)"  # title on next line (uppercase start)
        r"|"
        r"\n(\d+\.[\d.]*)\s+([A-ZČŠŽĆĐ][^\n]+)",  # title on same line
        re.MULTILINE,
    )

    raw_matches = []
    for m in header_pattern.finditer(full_text):
        # Determine which alternative matched
        if m.group(1) is not None:
            num = m.group(1).rstrip(".")
            title = m.group(2).strip()
        else:
            num = m.group(3).rstrip(".")
            title = m.group(4).strip()

        # Skip inline references: preceded by "poglavje", "poglavja", "poglavji"
        pre = full_text[max(0, m.start() - 30):m.start()]
        if re.search(r"poglavj[aei]?\s*$", pre, re.IGNORECASE):
            continue

        # Only keep the first occurrence of each section number
        if any(prev_num == num for prev_num, _, _ in raw_matches):
            continue

        raw_matches.append((num, title, m.start()))

    sections: Dict[str, Dict[str, str]] = {}
    for i, (num, title, pos) in enumerate(raw_matches):
        # Text starts after the title line
        title_end = full_text.index(title, pos) + len(title)
        next_pos = raw_matches[i + 1][2] if i + 1 < len(raw_matches) else len(full_text)
        section_text = full_text[title_end:next_pos].strip()
        sections[num] = {"title": title, "text": section_text}

    return sections


def _get_smpc(query: str) -> Dict[str, Any]:
    """Get SmPC (Summary of Product Characteristics) for a drug.

    Downloads the SmPC PDF from CBZ, extracts clinically relevant sections
    (indications, dosing, contraindications, interactions, side effects, etc.).

    USE THIS TOOL WHEN:
    - You need detailed clinical information about a drug
    - Looking up indications, dosing, contraindications, or interactions
    - Checking pregnancy/lactation safety
    - Getting pharmacological properties
    - Needing authoritative drug information from the SmPC document
    """
    if not query or len(query.strip()) < 2:
        return {"found": False, "error": "Query must be at least 2 characters"}

    try:
        search_query = f"([TXIMELAS1]=_{quote(query.strip(), safe='')}*)"
        url = f"{CBZ_API_BASE}?SearchView&Query={search_query}&SearchOrder=4&SearchMax=301"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text

        # Extract SmPC URLs and drug URLs from the same search results page
        smpc_map = _extract_smpc_urls(html)

        if not smpc_map:
            return {
                "found": False,
                "query": query,
                "error": "No SmPC document found for this drug",
            }

        # Use the first match
        drug_url = next(iter(smpc_map))
        smpc_url = smpc_map[drug_url]

        # Get drug name from the search results page
        soup = BeautifulSoup(html, "html.parser")
        drug_name = query
        for link in soup.find_all("a", href=re.compile(r"opendocument")):
            href = link.get("href", "").strip()
            if href.startswith("/"):
                href = f"https://www.cbz.si{href}"
            if href == drug_url:
                drug_name = link.get_text(strip=True)
                break

        # Fetch and parse PDF
        pdf_bytes = _fetch_smpc_pdf(smpc_url)
        if not pdf_bytes:
            return {
                "found": False,
                "query": query,
                "drug_name": drug_name,
                "smpc_url": smpc_url,
                "error": "Failed to download SmPC PDF",
            }

        all_sections = _parse_smpc_sections(pdf_bytes)

        # Filter to clinically relevant sections
        clinical_sections = {}
        for sec_num, sec_data in all_sections.items():
            if sec_num in SMPC_CLINICAL_SECTIONS:
                clinical_sections[sec_num] = sec_data

        return {
            "found": True,
            "query": query,
            "drug_name": drug_name,
            "smpc_url": smpc_url,
            "sections": clinical_sections,
        }

    except Exception as e:
        return {"found": False, "query": query, "error": str(e)[:200]}


def _extract_basic_info(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract drug name, ATC code, and prescribing regime from a CBZ drug page.

    CBZ uses textbarva0/textbarva01 cell pairs for labeled fields.
    """
    info = {"name": "", "atc": "", "regime": ""}

    title = soup.find("title")
    if title:
        info["name"] = title.get_text(strip=True)

    # Scan textbarva0 label cells paired with textbarva01 value cells
    for td in soup.find_all("td", class_="textbarva0"):
        label = td.get_text(strip=True)
        val_td = td.find_next_sibling("td", class_="textbarva01")
        if not val_td:
            continue
        value = val_td.get_text(strip=True)

        if "ATC" in label and not info["atc"]:
            info["atc"] = value
        if "režim" in label.lower() and not info["regime"]:
            info["regime"] = value

    return info


def _extract_classifications(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract classification sections (Razvrstitev zdravila) from a CBZ drug page.

    CBZ structure: textbarva02 section headers followed by rows of ts-7-1 (label) /
    ts-7-2 (value) cell pairs. The limitation text often appears in a separate row
    right after the "Omejitve predpis." row, with an empty ts-7-1 label.
    """
    classifications = []

    section_markers = soup.find_all("td", class_="textbarva02")

    for marker in section_markers:
        marker_text = marker.get_text(strip=True)

        list_type = None
        for keyword in ["Na listo ZZZS", "Na bolnišnični seznam", "Na seznam ampuliranih"]:
            if keyword.lower() in marker_text.lower():
                list_type = marker_text
                break

        if not list_type:
            continue

        classification = {
            "list_type": list_type,
            "lista": "",
            "limitation": "",
            "valid_from": "",
        }

        # Collect all ts-7-1/ts-7-2 pairs after this marker until the next
        # textbarva02 section header
        saw_omejitve = False
        for elem in marker.find_all_next("td", class_="ts-7-1"):
            # Stop if we've crossed into the next section
            # Check if there's a textbarva02 between the marker and this element
            prev_section = elem.find_previous("td", class_="textbarva02")
            if prev_section and prev_section != marker:
                break

            label = elem.get_text(strip=True)
            val_td = elem.find_next_sibling("td", class_="ts-7-2")
            value = val_td.get_text(strip=True) if val_td else ""

            if label.startswith("Lista"):
                classification["lista"] = value
                saw_omejitve = False
            elif "Omejitve" in label:
                # The value may be here or in the next row with empty label
                if value:
                    classification["limitation"] = value
                saw_omejitve = True
            elif saw_omejitve and not label and value:
                # Continuation row: limitation text with empty label
                classification["limitation"] = value
                saw_omejitve = False
            elif "Velja od" in label:
                classification["valid_from"] = value
                saw_omejitve = False
            else:
                saw_omejitve = False

        classifications.append(classification)

    return classifications


def _extract_prices(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract price information from a CBZ drug page.

    CBZ uses textbarva0/textbarva01 cell pairs for price fields like
    "Cena na debelo", "NPV", "Informativno doplačilo", "Vrsta zdravila".
    """
    prices = {
        "wholesale": "",
        "npv": "",
        "surcharge": "",
        "drug_type": "",
    }

    for td in soup.find_all("td", class_="textbarva0"):
        label = td.get_text(strip=True)
        val_td = td.find_next_sibling("td", class_="textbarva01")
        if not val_td:
            continue
        value = val_td.get_text(strip=True)

        if "Cena na debelo" in label:
            prices["wholesale"] = value
        elif label == "NPV :" or "najvišja prizn" in label.lower():
            prices["npv"] = value
        elif "doplačilo" in label.lower():
            prices["surcharge"] = value
        elif "Vrsta zdravila" in label:
            prices["drug_type"] = value

    return prices


def _fetch_drug_page_cached(url: str) -> Optional[str]:
    """Fetch a drug detail page HTML, using cache when available."""
    cache_key = f"drug_page_{hashlib.md5(url.encode()).hexdigest()}"
    if is_cache_fresh(cache_key, max_age_hours=DRUG_PAGE_CACHE_TTL):
        cached = read_cache(cache_key)
        if cached and isinstance(cached, list) and cached:
            return cached[0]  # stored as [html_string]

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
        write_cache(cache_key, [html])
        return html
    except Exception as e:
        print(f"Error fetching drug page {url}: {e}")
        return None


def _get_prescription_limitations(query: str) -> Dict[str, Any]:
    """Get prescription limitations and classification data for a drug from CBZ.

    Searches CBZ, fetches the drug detail pages, and extracts structured data about:
    - ZZZS list classification (Lista)
    - Prescribing limitations (Omejitve predpisovanja)
    - Prices (wholesale, NPV, surcharge)
    - ATC code and prescribing regime

    Limits to first 5 matching drugs to avoid excessive HTTP requests.
    """
    if not query or len(query.strip()) < 2:
        return {"found": False, "error": "Query must be at least 2 characters"}

    try:
        drug_urls = _search_drug_urls(query.strip())

        if not drug_urls:
            return {
                "found": False,
                "error": "No drugs found for your query",
                "query": query,
            }

        # Limit to first 5 results
        drug_urls = drug_urls[:5]

        results = []
        for name, url in drug_urls:
            html = _fetch_drug_page_cached(url)
            if not html:
                continue

            try:
                soup = BeautifulSoup(html, "html.parser")

                # Remove scripts/styles
                for tag in soup(["script", "style"]):
                    tag.decompose()

                basic_info = _extract_basic_info(soup)
                classifications = _extract_classifications(soup)
                prices = _extract_prices(soup)

                results.append({
                    "name": name,
                    "source": url,
                    "regime": basic_info["regime"],
                    "atc": basic_info["atc"],
                    "classifications": classifications,
                    "prices": prices,
                })
            except Exception as e:
                print(f"Error parsing drug page {url}: {e}")
                continue

        if not results:
            return {
                "found": False,
                "error": "Found drug URLs but failed to extract data",
                "query": query,
            }

        return {
            "found": True,
            "count": len(results),
            "query": query,
            "results": results,
        }
    except Exception as e:
        return {"found": False, "error": str(e)[:100]}


if __name__ == "__main__":
    # Example usage
    print("Testing drug search...")
    test_query = "Plivit D3"
    info = _get_drug_info(test_query)
    print(f"\nQuery: {test_query}")
    print(f"Found: {info['found']}")

    print("\nTesting prescription limitations...")
    test_query2 = "amoksiklav"
    limitations = _get_prescription_limitations(test_query2)
    print(f"\nQuery: {test_query2}")
    print(f"Found: {limitations['found']}")
    if limitations.get("results"):
        for r in limitations["results"]:
            print(f"  {r['name']}: {len(r['classifications'])} classifications")