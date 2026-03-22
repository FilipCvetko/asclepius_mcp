# FastMCP Server - Medical Tools
# Modular architecture: contacts, drugs, ICD-10, ZZZS, templates

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

from contacts import initialize_contacts, _get_phone_number
from drugs import initialize_drugs, _get_drug_info, _get_prescription_limitations
from icd10 import initialize_icd10, _get_icd10
from zzzs import initialize_zzzs, _get_zzzs_limitation, _browse_zzzs_rules, _list_zzzs_categories
from egradiva import initialize_egradiva, _search_egradiva
from templates import initialize_templates, _get_template, _list_templates, _format_note_prompt

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

mcp = FastMCP("medical-tools")

# Track module status for health check
_module_status = {}

# Initialize modules on startup (parallel)
print("Initializing medical tools...")
_init_start = time.time()

_init_functions = {
    "contacts": initialize_contacts,
    "drugs": initialize_drugs,
    "icd10": initialize_icd10,
    "zzzs": initialize_zzzs,
    "egradiva": initialize_egradiva,
    "templates": initialize_templates,
}

with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(fn): name for name, fn in _init_functions.items()}
    for future in as_completed(futures):
        name = futures[future]
        try:
            future.result()
            _module_status[name] = {"loaded": True, "error": None}
        except Exception as e:
            _module_status[name] = {"loaded": False, "error": str(e)}
            print(f"WARNING: {name} initialization failed: {e}")

_init_elapsed = round(time.time() - _init_start, 1)
print(f"Medical tools ready! ({_init_elapsed}s)")


# Register contact tools
@mcp.tool()
def get_phone_number(query: str) -> dict:
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
    return _get_phone_number(query)


# Register drug tools
@mcp.tool()
def get_drug_info(query: str) -> dict:
    """Get drug information and usage details.

    Search for drugs by name, indication, or usage pattern.
    Returns all relevant drugs with similarity scores.

    USE THIS TOOL WHEN:
    - Looking up drug information by name
    - Finding drugs for a specific condition
    - Getting usage and dosage information
    - Getting packaging information and prices
    - Searching for drug alternatives
    - Searching for prescribing limitations by the insurance company - ZZZS

    """

    return _get_drug_info(query)


# Register ICD-10 tools
@mcp.tool()
def get_icd10_code(query: str) -> dict:
    """Look up MKB-10/ICD-10 diagnostic codes. Search by description (Slovenian or English)
    to find codes, or enter a code to get its full description. Bidirectional.

    USE THIS TOOL WHEN:
    - Looking up a diagnostic code by description (e.g. "glavobol", "headache")
    - Looking up what a code means (e.g. "J06.9", "I10")
    - Finding the right code for a diagnosis
    - Searching for related diagnostic codes

    EXAMPLES:
    • "J06.9" → Akutna okužba zgornjih dihal, neopredeljena
    • "glavobol" → R51 Glavobol
    • "diabetes" → E10-E14 codes
    • "I10" → Esencialna (primarna) hipertenzija
    """
    return _get_icd10(query)


# Register ZZZS tools
@mcp.tool()
def get_zzzs_drug_limitation(query: str) -> dict:
    """Look up ZZZS prescribing limitations, list status, and reimbursement rules for a drug.

    USE THIS TOOL WHEN:
    - Checking if a drug has prescribing limitations (omejitve predpisovanja)
    - Finding out which ZZZS list a drug is on (pozitivna/intermediarna lista)
    - Checking reimbursement status and NPV pricing
    - Looking up ATC codes and therapeutic groups

    EXAMPLES:
    • "amoksicilin" → Prescribing limitations for amoxicillin
    • "pantoprazol" → List status and limitations for pantoprazole
    • "atorvastatin" → Statin prescribing rules
    """
    return _get_zzzs_limitation(query)



@mcp.tool()
def browse_zzzs_rules(category: str) -> dict:
    """Browse ZZZS rules by category — returns all articles in a topic area.

    Each result includes a source_url linking to the original PDF document.

    USE THIS TOOL WHEN:
    - You want to see ALL rules in a specific area (not just search)
    - Browsing a whole chapter or section of the Pravila OZZ
    - Getting a comprehensive view of rules for a category

    CATEGORIES (use any of these):
    zdravila, pripomočki, zobozdravstvo, napotnice specialist,
    predpisovanje zdravil, bolniška, prevoz, nujna pomoč,
    specialistično, rehabilitacija, zdravilišče, spremstvo,
    nadomestilo plače, potni stroški, izbira zdravnika,
    postopek osnovno, postopek tujina, splošne določbe, ...

    EXAMPLES:
    • "pripomočki" → All medical device articles (V/*)
    • "zdravila" → Drug coverage articles (IV/8)
    • "napotnice specialist" → Referral procedure articles (XIII/6)
    """
    return _browse_zzzs_rules(category)


@mcp.tool()
def list_zzzs_categories() -> dict:
    """List all available ZZZS rule categories with article counts.

    USE THIS TOOL WHEN:
    - You want to know what topics/categories of rules are available
    - Before browsing, to see the full list of areas covered
    - Understanding the scope of ZZZS Pravila OZZ
    """
    return _list_zzzs_categories()


# Register e-gradiva search tools
@mcp.tool()
def search_zzzs_documents(query: str, category: str = None) -> dict:
    """Semantic search across ALL ZZZS e-gradiva documents (~766 documents, 33 categories).

    Searches circulars, legal acts, billing protocols, device standards, clinical
    guidelines, agreements, Pravila OZZ articles, and more. Each result includes
    a clickable source reference (source_url + page_number) for verification.

    USE THIS TOOL WHEN:
    - Looking up ZZZS rules, circulars, or guidelines on any topic
    - Finding billing/accounting protocols and procedures
    - Searching for medical device standards and specifications
    - Looking for agreements between ZZZS and healthcare providers
    - Researching any ZZZS regulation or policy document
    - When browse_zzzs_rules doesn't cover the topic (it only has Pravila OZZ)

    EXAMPLES:
    • "napotnica za specialista" → Referral procedures and rules
    • "bolniški stalež" → Sick leave rules and procedures
    • "obračun storitev" → Billing and accounting rules
    • "medicinski pripomočki standardi" → Medical device standards
    • "zobozdravstvo obračun" → Dental billing procedures

    Optional: filter by category name to narrow results.
    """
    return _search_egradiva(query, category=category)


@mcp.tool()
def poisci_zzzs_dokumente(query: str, category: str = None) -> dict:
    """Semantično iskanje po VSEH ZZZS e-gradivih (~766 dokumentov, 33 kategorij).

    Slovensko poimenovanje orodja search_zzzs_documents — deluje enako.
    Išče po okrožnicah, pravnih aktih, obračunskih pravilih, standardih,
    kliničnih smernicah, dogovorih in več. Vrne izvorne URL-je in številke strani.
    """
    return _search_egradiva(query, category=category)


# Register template tools
@mcp.tool()
def get_note_template(template_name: str) -> dict:
    """Get the structure and sections of a clinical note template.

    USE THIS TOOL WHEN:
    - Looking up available note templates
    - Getting the structure of a specific template
    - Checking what sections a template includes
    """
    return _get_template(template_name)


# Register template prompt
@mcp.prompt()
def format_clinical_note(template_name: str, raw_text: str) -> str:
    """Format dictated clinical notes into structured templates (Anamneza, Status, Terapija, etc.)

    Use this prompt to format raw dictated clinical text into a properly structured note
    using one of the available templates.
    """
    return _format_note_prompt(template_name, raw_text)


# Register template resource
@mcp.resource("templates://list")
def list_templates() -> str:
    """List all available clinical note templates."""
    return _list_templates()


# Register prescription limitations tool
@mcp.tool()
def get_prescription_limitations(query: str) -> dict:
    """Get structured prescription limitation data for a drug from CBZ.

    Searches CBZ and extracts ZZZS classification, prescribing limitations,
    prices, ATC codes, and prescribing regime from drug detail pages.
    More targeted than get_drug_info — returns structured data instead of full page text.

    USE THIS TOOL WHEN:
    - Checking ZZZS list classification (P, PC, PC*, etc.)
    - Looking up specific prescribing limitations (omejitve predpisovanja)
    - Getting drug prices, NPV, and surcharges
    - Finding ATC codes and prescribing regime

    EXAMPLES:
    • "amoksiklav" → Classification, limitations, prices for Amoksiklav
    • "pantoprazol" → List status and limitations for pantoprazole
    • "rosuvastatin" → Statin classification and limitations
    """
    return _get_prescription_limitations(query)


# ── Health check ─────────────────────────────────────────────────────

@mcp.tool()
def health_check() -> dict:
    """Check the status of all medical tool modules.

    Returns which modules loaded successfully, record counts, and any warnings.
    Use this to diagnose issues when a tool returns unexpected errors.
    """
    from contacts import CONTACTS
    from icd10 import ICD10_CODES
    from zzzs import ZZZS_DRUGS, ZZZS_GROUPS, ZZZS_RULES
    from egradiva import COLLECTION
    from templates import TEMPLATES

    modules = {
        "contacts": {"loaded": bool(CONTACTS), "records": len(CONTACTS) if CONTACTS else 0},
        "drugs": {"loaded": True, "records": "live API (CBZ)"},
        "icd10": {"loaded": bool(ICD10_CODES), "records": len(ICD10_CODES) if ICD10_CODES else 0},
        "zzzs": {"loaded": bool(ZZZS_DRUGS), "records": f"{len(ZZZS_DRUGS)} drugs, {len(ZZZS_GROUPS)} groups, {len(ZZZS_RULES)} rules"},
        "egradiva": {"loaded": COLLECTION is not None, "records": COLLECTION.count() if COLLECTION else 0},
        "templates": {"loaded": bool(TEMPLATES), "records": len(TEMPLATES) if TEMPLATES else 0},
    }

    for name, status in modules.items():
        init = _module_status.get(name, {})
        status["init_error"] = init.get("error")

    return {
        "status": "ok" if all(m["loaded"] for m in modules.values()) else "degraded",
        "startup_time_s": _init_elapsed,
        "modules": modules,
    }


@mcp.tool()
def preveri_stanje() -> dict:
    """Preveri stanje vseh modulov medicinskih orodij.

    Slovensko poimenovanje orodja health_check — deluje enako.
    """
    return health_check()


# ── Slovenian aliases ──────────────────────────────────────────────────

@mcp.tool()
def poisci_telefonsko(query: str) -> dict:
    """Poišči telefonske številke medicinskih ustanov in kontaktov.

    Slovensko poimenovanje orodja get_phone_number — deluje enako.
    """
    return _get_phone_number(query)


@mcp.tool()
def poisci_zdravilo(query: str) -> dict:
    """Poišči informacije o zdravilih v CBZ bazi (Centralna baza zdravil).

    Slovensko poimenovanje orodja get_drug_info — deluje enako.
    """
    return _get_drug_info(query)


@mcp.tool()
def poisci_mkb10(query: str) -> dict:
    """Poišči MKB-10 diagnostične kode ali opise diagnoz.

    Slovensko poimenovanje orodja get_icd10_code — deluje enako.
    """
    return _get_icd10(query)


@mcp.tool()
def poisci_omejitve_zzzs(query: str) -> dict:
    """Poišči omejitve predpisovanja ZZZS za zdravilo.

    Slovensko poimenovanje orodja get_zzzs_drug_limitation — deluje enako.
    """
    return _get_zzzs_limitation(query)



@mcp.tool()
def prebrskaj_pravila_zzzs(category: str) -> dict:
    """Prebrskaj pravila ZZZS po kategoriji — vrne vse člene v izbranem področju.

    Slovensko poimenovanje orodja browse_zzzs_rules — deluje enako.
    """
    return _browse_zzzs_rules(category)


@mcp.tool()
def seznam_kategorij_zzzs() -> dict:
    """Seznam vseh kategorij pravil ZZZS s številom členov.

    Slovensko poimenovanje orodja list_zzzs_categories — deluje enako.
    """
    return _list_zzzs_categories()


@mcp.tool()
def poisci_predlogo(template_name: str) -> dict:
    """Poišči strukturo predloge za klinični zapis.

    Slovensko poimenovanje orodja get_note_template — deluje enako.
    """
    return _get_template(template_name)


@mcp.tool()
def poisci_omejitve_predpisovanja(query: str) -> dict:
    """Poišči omejitve predpisovanja zdravila iz CBZ strani.

    Slovensko poimenovanje orodja get_prescription_limitations — deluje enako.
    """
    return _get_prescription_limitations(query)


if __name__ == "__main__":
    import os
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    if transport in ("streamable-http", "sse"):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route, Mount

        async def _health(request):
            return JSONResponse({"status": "ok"})

        mcp_app = mcp.http_app(transport=transport)
        app = Starlette(routes=[
            Route("/health", _health),
            Mount("/", app=mcp_app),
        ])

        import uvicorn
        uvicorn.run(app, host=host, port=port)
    else:
        mcp.run()
