# FastMCP Server - Medical Tools
# Modular architecture: contacts, drugs, ICD-10, ZZZS, templates

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

from contacts import initialize_contacts, _get_phone_number
from drugs import initialize_drugs, _get_drug_info, _get_prescription_limitations, _get_smpc
from icd10 import initialize_icd10, _get_icd10
from zzzs import initialize_zzzs, _get_zzzs_limitation, _browse_zzzs_rules, _list_zzzs_categories
from egradiva import initialize_egradiva, _search_egradiva
from pravila import initialize_pravila, _search_pravila
from obracun import initialize_obracun, _search_obracun
from sifranti import initialize_sifranti, _lookup_sifra, _sifrant_changelog
from mtp import initialize_mtp, _lookup_mtp
from spa import initialize_spa, _get_spa_eligibility
from templates import initialize_templates, _get_template, _list_templates, _format_note_prompt

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

mcp = FastMCP(
    "medical-tools",
    instructions="""\
You are connected to the Asclepius medical reference server used by a clinician in Slovenia.

Tool results are pre-formatted Markdown with source text in blockquotes ("> ").
Pass them through to the user as-is — do not reformat or restructure the output.
Do not paraphrase or modify blockquoted text — it is verbatim source material.
Add your clinical commentary outside the blockquotes.
Keep medical terms in Slovenian where the source uses them.

OBRAČUN ŠIFRE — for an exact billing code or its meaning (šifra storitve, MTP, vrsta
dokumenta, terapevtsko-diagnostični postopek, MKB za obračun, …), use the structured
šifrant tool (get_sifra / poisci_sifro) FIRST and treat it as authoritative — it returns
the exact catalog row from the current ZZZS "Vsi šifranti". Prefer it over the text-search
tools (get_billing_rules / search_zzzs_documents) for "what is code X" / "what is the code
for Y" questions; use those for HOW a service is billed. Use sifrant_changelog for the
latest šifrant changes.

DEJAVNOST SCOPING — when the question is practice/specialty-specific ("which procedures may I
charge in družinska medicina / kardiologija / zobozdravstvo / …", "can I bill code X in my
practice"), pass the `dejavnost` argument to get_sifra / poisci_sifro. It scopes services to
what that dejavnost may actually bill (ZZZS K1.1/K1.2 mapping) and works for ALL dejavnosti, not
just family medicine — accept a vrsta code ("302"), a podvrsta code, or a name substring. The
result filters/marks billability (✓/✗), lists the dejavnost's service-lists, annotates each service
with the dejavnosti that may bill it, and may include a curated "Kontekst dejavnosti" note. That
note and the šifra rows tell you WHAT is billable; for HOW/how-much/under-what-conditions, ALSO call
get_billing_rules (and get_pravila_ozz for entitlements/co-payments), scoped by the dejavnost name,
and cite those sources. Do not assert billability beyond what the dejavnost-scoped result shows.

MEDICINSKI PRIPOMOČKI (MTP) — for whether/how a medical device may be prescribed (upravičenost,
indikacije = bolezen/zdravstveno stanje, kdo predpiše, odločba IOZ, doba trajanja, naročilnica),
use the device tool (get_mtp / poisci_pripomocek) — authoritative ZZZS device list. It resolves a
device code, a device name (e.g. bergla, voziček), OR a patient condition (e.g. amputacija) to the
eligible device(s) with their prescribing rules. Prefer it over the text-search tools for device
eligibility/indication questions.

SOURCE ATTRIBUTION — when you report information that came from a document-search tool
(search_zzzs_documents, get_pravila_ozz, get_billing_rules, and any result that includes a
"Vir:" link), always cite its source inline, next to the claim it supports: the document
title, the page ("str. N") if given, and the clickable source URL. Each result already
carries this as a "**Vir:** [...](url)" line — keep it visible so the clinician can verify.
Cite ONLY sources/URLs that appear in the tool result; never invent, guess, shorten, or
reconstruct a URL. If a result has no source link, attribute it to the named document/tool.
""",
)

# Track module status for health check
_module_status = {}
_init_elapsed = None  # set when background init finishes (None = still initializing)
_EMBED = None         # shared embedding fn + ChromaDB client (set in _initialize_all),
_CLIENT = None        # reused by the periodic refresh so there's only one client per path

import os as _os
import shutil as _shutil
import threading as _threading
import chromadb as _chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_chroma_path = _os.environ.get("CHROMADB_PATH", str(Path(__file__).parent / "data" / "chromadb"))
_SEED_DIR = "/seed/chromadb"


def _read_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _seed_chromadb_if_needed():
    """Copy the image-baked ChromaDB (/seed) onto the writable volume path if it's
    missing or stale. ChromaDB's Rust/sqlite bindings need a real filesystem (the
    volume) — the image overlay (/seed) times out — so we read from /seed but serve
    from CHROMADB_PATH. Versioned by data/chromadb/.seed_version."""
    if _chroma_path == _SEED_DIR or not _os.path.isdir(_SEED_DIR):
        return
    seed_ver = _read_file(_os.path.join(_SEED_DIR, ".seed_version"))
    # A completion marker written LAST = the copy fully finished. If a seed is interrupted
    # (e.g. machine restart mid-copy), the marker is absent/stale and we re-seed — avoiding
    # a half-written, "malformed" chroma.sqlite3.
    done_marker = _os.path.join(_chroma_path, ".seed_complete")
    if _read_file(done_marker) == seed_ver and seed_ver is not None:
        return  # already fully seeded for this version
    print(f"Seeding ChromaDB {_SEED_DIR} -> {_chroma_path} (target {seed_ver})...")
    _shutil.rmtree(_chroma_path, ignore_errors=True)
    _shutil.copytree(_SEED_DIR, _chroma_path)
    with open(done_marker, "w") as f:
        f.write(seed_ver or "")
    print("ChromaDB seed complete.")


def _initialize_all():
    """Heavy startup (seed + model + collections) — runs in a background thread so the
    HTTP server binds and passes health checks immediately, instead of blocking on the
    ~1-3 min of seeding/model-loading. ChromaDB tools simply report 'not loaded' until
    this finishes (a brief window after boot)."""
    global _init_elapsed, _EMBED, _CLIENT
    t0 = time.time()
    print("Initializing medical tools (background)...")
    try:
        _seed_chromadb_if_needed()
    except Exception as e:
        print(f"WARNING: ChromaDB seed failed: {e}")

    # One shared embedding model + one shared ChromaDB client for all ChromaDB-backed
    # modules — ChromaDB allows only one PersistentClient per path, and 3× model loads
    # are wasteful. Created here (not at import) so a failure degrades, not crashes.
    # Stored as globals so the periodic refresh (refresh.py) reuses the SAME client/model.
    embed = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    client = _chromadb.PersistentClient(path=_chroma_path)
    _EMBED, _CLIENT = embed, client

    init_functions = {
        "contacts": initialize_contacts,
        "drugs": initialize_drugs,
        "icd10": initialize_icd10,
        "zzzs": initialize_zzzs,
        "egradiva": lambda: initialize_egradiva(embed, client),
        "pravila": lambda: initialize_pravila(embed, client),
        "obracun": lambda: initialize_obracun(embed, client),
        "sifranti": initialize_sifranti,  # local JSON catalog, no ChromaDB dependency
        "mtp": initialize_mtp,            # local JSON catalog (medical-device prescribing rules)
        "spa": initialize_spa,
        "templates": initialize_templates,
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fn): name for name, fn in init_functions.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                _module_status[name] = {"loaded": True, "error": None}
            except Exception as e:
                _module_status[name] = {"loaded": False, "error": str(e)}
                print(f"WARNING: {name} initialization failed: {e}")

    _init_elapsed = round(time.time() - t0, 1)
    print(f"Medical tools ready! ({_init_elapsed}s)")

    _start_refresh_scheduler()


# ── Periodic data refresh (incremental, atomic, never crashes serving) ────────
_refresh_lock = _threading.Lock()
_refresh_running = False


def _run_refresh(manual: bool = False) -> dict:
    """Run one refresh under a lock (skips if one is already in progress). Never raises."""
    global _refresh_running
    if not _refresh_lock.acquire(blocking=False):
        return {"status": "skipped", "reason": "a refresh is already running"}
    try:
        _refresh_running = True
        import refresh
        return refresh.refresh_all(embed=_EMBED, client=_CLIENT, manual=manual)
    except Exception as e:  # defensive — refresh_all already guards, but never let this crash
        print(f"WARNING: refresh crashed: {e}")
        return {"status": "error", "fatal": str(e)}
    finally:
        _refresh_running = False
        _refresh_lock.release()


def _refresh_loop():
    """Every REFRESH_INTERVAL_DAYS run a refresh; catch up on boot if overdue. Hourly wake-check."""
    interval = float(_os.environ.get("REFRESH_INTERVAL_DAYS", "2")) * 86400
    import datetime as _dt
    import refresh
    while True:
        try:
            last = (refresh.get_status() or {}).get("last_refresh") or {}
            ts = last.get("finished_at") or last.get("started_at")
            if not ts:
                # First boot ever: seed the clock so the next auto-run is ~one interval out (don't
                # hammer on every deploy). Immediate verification is via the manual refresh_now.
                refresh._save_json(refresh.STATE_FILE,
                                   {"last_refresh": {"finished_at": refresh._now(), "status": "seed"},
                                    "history": []})
            else:
                age = (_dt.datetime.now(_dt.timezone.utc)
                       - _dt.datetime.fromisoformat(ts)).total_seconds()
                if age >= interval:
                    print("refresh: scheduled run starting")
                    _run_refresh(manual=False)
        except Exception as e:
            print(f"refresh loop error (continuing): {e}")
        time.sleep(min(interval, 3600))   # re-check at most hourly


def _start_refresh_scheduler():
    if _os.environ.get("REFRESH_DISABLED") == "1":
        print("refresh: scheduler disabled (REFRESH_DISABLED=1)")
        return
    _threading.Thread(target=_refresh_loop, daemon=True).start()
    print("refresh: scheduler started")


def _start_background_init():
    """Kick off heavy init in a background thread (called once the server is about to
    start) so uvicorn binds and /health passes before the seed/model/collections load."""
    _threading.Thread(target=_initialize_all, daemon=True).start()


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


@mcp.tool()
def get_smpc(query: str) -> dict:
    """Get the SmPC (Summary of Product Characteristics) for a drug.

    Downloads the official SmPC PDF from CBZ and extracts clinically relevant
    sections: indications, dosing, contraindications, interactions, warnings,
    pregnancy/lactation, adverse effects, overdose, and pharmacology.

    USE THIS TOOL WHEN:
    - You need detailed clinical information about a drug
    - Looking up official indications, dosing, or contraindications
    - Checking drug interactions or pregnancy safety
    - Getting pharmacokinetic/pharmacodynamic properties
    - Needing authoritative drug information beyond basic CBZ listing

    EXAMPLES:
    • "amoksiklav" → Full SmPC sections for Amoksiklav
    • "ibuprofen" → Indications, dosing, interactions for Ibuprofen
    • "pantoprazol" → Detailed clinical data for pantoprazole
    """
    return _get_smpc(query)


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


# Register Pravila OZZ tools (authoritative rules of mandatory health insurance)
@mcp.tool()
def get_pravila_ozz(query: str) -> dict:
    """Authoritative search of the Pravila OZZ — the Rules of Mandatory Health Insurance
    (Pravila obveznega zdravstvenega zavarovanja, NPB42 consolidated text).

    Use THIS tool for any question about what the mandatory health insurance (OZZ)
    actually entitles, covers, or reimburses, and the procedures around it. It searches
    ONLY the 307 official Pravila OZZ articles — not the broad e-gradiva corpus — so the
    answer comes from the governing legal text and cites the exact article number.

    USE THIS TOOL WHEN:
    - A provider asks about coverage / reimbursement / entitlements under OZZ
    - Rules on medical devices (medicinski pripomočki), their standards or durability
    - Referrals (napotnice), choice/change of doctor, specialist procedures
    - Sick leave (bolniška, nadomestilo plače), travel costs (potni stroški), escort (spremstvo)
    - Dental care, spa/rehabilitation (zdravilišče, rehabilitacija) entitlements
    - Treatment abroad, drug/aid prescribing procedures under the Pravila
    - Looking up a specific article (e.g. "23. člen")

    Returns the governing article(s) verbatim with section, category, cross-referenced
    articles, and the official source link. Prefer this over search_zzzs_documents for
    rule/entitlement questions — that tool spans mixed, partly superseded material.

    EXAMPLES:
    • "veljavnost napotnice" → referral validity articles
    • "povračilo potnih stroškov" → travel cost reimbursement
    • "doplačilo za zobni most" → dental prosthetics co-payment
    • "23. člen" → exact article 23
    """
    return _search_pravila(query)


@mcp.tool()
def poisci_pravila_ozz(query: str) -> dict:
    """Poišči po Pravilih obveznega zdravstvenega zavarovanja (Pravila OZZ, NPB42).

    Slovensko poimenovanje orodja get_pravila_ozz — deluje enako. Verodostojen vir za
    vprašanja o pravicah, kritju in povračilih iz obveznega zdravstvenega zavarovanja;
    išče samo po uradnih členih Pravil OZZ in navede točno številko člena.
    """
    return _search_pravila(query)


# Register billing/coding rules tool (obračun ambulantnih storitev)
@mcp.tool()
def get_billing_rules(query: str) -> dict:
    """Authoritative rules for billing/coding OUTPATIENT health services (obračun, šifriranje
    storitev) under mandatory health insurance — current contract year.

    Searches ONLY the current ZZZS billing sources (Vsebinsko navodilo o beleženju in
    obračunavanju, Uredba o programih storitev, Standardi kodiranja, billing Q&A) — isolated
    from the broad e-gradiva corpus — so answers come from the documents that actually govern
    how a provider records and bills a service.

    USE THIS TOOL WHEN:
    - How to bill/record a specific service or visit (prvi/ponovni pregled, posvet, poseg)
    - Point values (točke) and how a service is valued
    - Constraints: who may bill it, frequency limits, which services may/may not be billed together
    - What a billing/service code means and the conditions for using it
    - Coding standards (standardi kodiranja) for diagnoses/procedures

    Scope note: v1 covers OUTPATIENT billing; hospital SPP is not yet included. Billing rules
    change yearly — results carry the source document's date; verify the contract year.

    EXAMPLES:
    • "kako se obračuna prvi pregled pri specialistu"
    • "ali se lahko dve storitvi obračunata na isti dan"
    • "kdo lahko zaračuna storitev X"
    • "standardi kodiranja glavna diagnoza"
    """
    return _search_obracun(query)


@mcp.tool()
def poisci_pravila_obracuna(query: str) -> dict:
    """Poišči pravila za obračun in šifriranje ambulantnih zdravstvenih storitev (ZZZS).

    Slovensko poimenovanje orodja get_billing_rules — deluje enako. Išče samo po aktualnih
    dokumentih za obračun (Vsebinsko navodilo, Uredba o programih storitev, Standardi
    kodiranja, vprašanja in odgovori); obračunska pravila se letno spreminjajo.
    """
    return _search_obracun(query)


# Register structured šifrant lookup (authoritative obračun codes — "Vsi šifranti" XML)
@mcp.tool()
def get_sifra(query: str, sifrant: str = "", dejavnost: str = "") -> dict:
    """Authoritative, EXACT lookup of ZZZS obračun šifre (billing codes) and their meaning,
    from the current "Vsi šifranti" catalog (full set, ~96 code-lists, parsed from ZZZS XML).

    This is the priority source for code↔description questions — it returns the exact catalog
    row (with validity dates), not free text. Use it BEFORE get_billing_rules /
    search_zzzs_documents for "what is code X" or "what is the code for Y". Bidirectional:
    pass a code → get its description + fields; pass a description → get matching codes.

    Covers (among others): šifrant 15 storitve (service codes), 50.1 MKB (diagnoze za obračun),
    15.40 vrste MTP (medicinsko-tehnični pripomočki), 38.11 terapevtsko-diagnostični postopki,
    26 vrste dokumentov, 37 tuji nosilci, plus dozens of smaller code-lists.

    DEJAVNOST SCOPING (key for "what may I bill in my practice"): pass `dejavnost` to scope service
    answers to what a given specialty/practice (vrsta zdravstvene dejavnosti) may actually bill —
    derived from the ZZZS K1.1/K1.2 mapping (dovoljene storitve/seznami po dejavnostih). Works for
    ALL dejavnosti, not just family medicine. Effects:
      • description query + dejavnost → results filtered to that dejavnost's billable services;
      • code query + dejavnost → marks whether that code is billable there (✓/✗);
      • empty query + dejavnost → lists the services that dejavnost may bill.
    Every service result is also annotated with the dejavnosti that may bill it. For HOW/how-much/
    conditions, still cross-reference get_billing_rules (Splošni dogovor / Vsebinsko navodilo).

    Args:
        query: a šifra (e.g. "AA001", "B83B", "0505") OR a description ("prvi pregled",
               "voziček na ročni pogon"). Leave empty with `dejavnost` set to list billable services.
        sifrant: optional filter to one šifrant — its oznaka (e.g. "15", "50.1") or a name
                 substring (e.g. "storitev", "MTP"). Leave empty to search all.
        dejavnost: optional specialty/practice scope — a vrsta code (e.g. "302", "111", "404"),
                 a podvrsta code, or a name substring (e.g. "družinska", "kardiolog", "zobozdrav",
                 "pediatr"). Leave empty for no dejavnost scoping.

    EXAMPLES:
    • "AA001" → exact service-code row (šifrant 15)
    • "prvi pregled", sifrant="15" → service codes whose description matches
    • "0505" → MTP vrsta (voziček na ročni pogon aktivni)
    • "laboratorij", dejavnost="družinska medicina" → only FM-billable lab services
    • "", dejavnost="kardiologija" → list the services cardiology may bill
    • "AA001", dejavnost="zobozdravstvo" → marks AA001 as NOT billable in dentistry (✗)
    """
    return _lookup_sifra(query, sifrant=sifrant or None, dejavnost=dejavnost or None)


@mcp.tool()
def poisci_sifro(query: str, sifrant: str = "", dejavnost: str = "") -> dict:
    """Poišči obračunsko šifro ZZZS in njen pomen v aktualnem šifrantu "Vsi šifranti".

    Slovensko poimenovanje orodja get_sifra — deluje enako. Verodostojen, točen vir za šifre
    obračuna (storitve, MKB, MTP, terapevtsko-diagnostični postopki, vrste dokumentov …):
    vrne točno vrstico iz šifranta. Dvosmerno: šifra → opis, ali opis → šifra. Uporabi PRED
    iskanjem po besedilu (poisci_pravila_obracuna / poisci_zzzs_dokumente) za vprašanja
    "kaj pomeni šifra X" oz. "katera šifra za Y". `sifrant` (npr. "15" ali "storitev") zoži iskanje.

    `dejavnost` zameji storitve na tisto, kar sme obračunati dana dejavnost (vrsta zdravstvene
    dejavnosti) — po mapiranju ZZZS K1.1/K1.2. Velja za VSE dejavnosti, ne le družinsko medicino:
    šifra (npr. "302", "111", "404"), šifra podvrste ali del imena ("družinska", "kardiolog",
    "zobozdrav"). Opis + dejavnost → zamejen nabor; šifra + dejavnost → oznaka ✓/✗ obračunljivosti;
    prazen query + dejavnost → seznam obračunljivih storitev dejavnosti. Za pogoje/točke dodaj
    poisci_pravila_obracuna.
    """
    return _lookup_sifra(query, sifrant=sifrant or None, dejavnost=dejavnost or None)


@mcp.tool()
def sifrant_changelog() -> dict:
    """Latest changes to the ZZZS obračun šifranti — the change summary (vsebina sprememb) of
    the most recent "Vsi šifranti" publication (objava leto/zaporedna št. + datum).

    Use when the clinician asks what changed in the šifranti / latest publication.
    """
    return _sifrant_changelog()


# Register medical-device (MTP) prescribing tool — authoritative ZZZS device list
@mcp.tool()
def get_mtp(query: str) -> dict:
    """Authoritative prescribing rules for MEDICAL DEVICES (medicinsko-tehnični pripomočki, MTP)
    from the current ZZZS "Seznam medicinskih pripomočkov s šifrantom, MK, postopki".

    For a device this returns the structured rules a clinician needs to prescribe it, above all
    the **indications** (bolezen / zdravstveno stanje in drugi pogoji that justify it), plus who
    may prescribe (pristojnost), whether an odločba imenovanega zdravnika is required, doba
    trajanja, naročilnica rules, izposoja and the price standard. This is the priority source for
    device eligibility — use it BEFORE the text-search tools for such questions.

    Bidirectional — the query may be:
      • a device code (e.g. "0501"),
      • a device name (e.g. "bergla", "voziček", "slušni aparat"), or
      • a patient condition (e.g. "amputacija", "pareza") → returns the eligible device(s).

    EXAMPLES:
    • "bergla" → crutch rows with indications (pareza, ankiloza, amputacija…) + who may prescribe
    • "0501" → exact device row
    • "amputacija nadlakti" → matching prostheses and their conditions
    """
    return _lookup_mtp(query)


@mcp.tool()
def poisci_pripomocek(query: str) -> dict:
    """Poišči pravila za predpis medicinsko-tehničnega pripomočka (MTP) iz aktualnega ZZZS
    "Seznama medicinskih pripomočkov s šifrantom, MK, postopki".

    Slovensko poimenovanje orodja get_mtp — deluje enako. Verodostojen vir za upravičenost in
    pogoje predpisa pripomočka: vrne **indikacije** (bolezen/zdravstveno stanje in drugi pogoji),
    kdo predpiše (pristojnost), ali je potrebna odločba IOZ, dobo trajanja, pravila naročilnice,
    izposojo in cenovni standard. Vprašanje je lahko šifra (npr. "0501"), ime pripomočka (npr.
    "bergla", "voziček") ali bolezensko stanje (npr. "amputacija") → vrne ustrezne pripomočke.
    Uporabi PRED iskanjem po besedilu za vprašanja o upravičenosti/indikacijah pripomočkov.
    """
    return _lookup_mtp(query)


# Register spa eligibility tools
@mcp.tool()
def get_spa_eligibility(query: str) -> dict:
    """Check which Slovenian spas (zdravilišča) are eligible for rehabilitation
    under mandatory health insurance (ZZZS) based on disease type or spa name.

    Query can be a disease/condition description, a standard type (tip 1-9),
    or a spa name. Returns matching spas with eligibility level (A/B).

    USE THIS TOOL WHEN:
    - Checking which spas cover a specific condition for rehab
    - Finding out if a spa is eligible for a patient's diagnosis
    - Looking up rehabilitation options under ZZZS insurance

    STANDARD TYPES (tip 1-9):
    • tip 1: Vnetne revmatske bolezni
    • tip 2: Degenerativni izvensklepni revmatizem
    • tip 3: Stanje po poškodbah in operacijah na lokomotornem sistemu
    • tip 4: Nevrološke bolezni, CVI, živčno-mišične bolezni
    • tip 5: Bolezni ter stanja po operacijah srca in ožilja
    • tip 6: Ginekološke bolezni, stanja po op. v mali medenici
    • tip 7: Kožne bolezni
    • tip 8: Gastroenterološke in endokrine bolezni
    • tip 9: Obolenja dihal

    EXAMPLES:
    • "tip 1" → All spas for inflammatory rheumatic diseases
    • "Terme Čatež" → All standards covered by that spa
    • "kožne bolezni" → Spas for skin diseases (tip 7)
    • "nevrološke" → Spas for neurological rehab (tip 4)
    """
    return _get_spa_eligibility(query)


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


# ── Periodic refresh tools ───────────────────────────────────────────

@mcp.tool()
def refresh_now() -> dict:
    """Manually trigger a data refresh now (incremental: only new/changed ZZZS docs are pulled +
    embedded; šifranti/MTP catalogs re-fetched if newer). Runs in the background server process,
    never re-embeds everything, and emails a summary. Use this for the post-deploy test run.

    Returns the refresh report (status, what changed, counts). If a refresh is already running,
    returns status 'skipped'.
    """
    if _refresh_running:
        return {"status": "skipped", "reason": "a refresh is already running"}
    # Run in the background (a run can take minutes) so the call returns immediately; the result
    # arrives by email and via refresh_status.
    _threading.Thread(target=lambda: _run_refresh(manual=True), daemon=True).start()
    return {"status": "started",
            "note": "Refresh running in background — you'll get an email summary; poll refresh_status."}


@mcp.tool()
def osvezi_zdaj() -> dict:
    """Ročno sproži osvežitev podatkov ZZZS (inkrementalno — samo novi/spremenjeni dokumenti).

    Slovensko poimenovanje orodja refresh_now — deluje enako; pošlje povzetek po e-pošti.
    """
    return refresh_now()


@mcp.tool()
def refresh_status() -> dict:
    """Status of the periodic data refresh: last run time, what changed (šifrant objava, MTP date,
    new/pruned e-gradiva docs), and recent history."""
    import refresh
    st = refresh.get_status()
    return {"running": _refresh_running,
            "interval_days": float(_os.environ.get("REFRESH_INTERVAL_DAYS", "2")),
            "last_refresh": st.get("last_refresh"),
            "history": st.get("history", [])[:5]}


@mcp.tool()
def stanje_osvezitve() -> dict:
    """Stanje samodejne osvežitve podatkov (zadnji zagon, kaj se je spremenilo).

    Slovensko poimenovanje orodja refresh_status — deluje enako.
    """
    return refresh_status()


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
    from pravila import COLLECTION as PRAVILA_COLLECTION, RULES as PRAVILA_RULES
    from obracun import COLLECTION as OBRACUN_COLLECTION
    from sifranti import (SIFRANTI as SIFRANT_CAT, _ROWS as SIFRANT_ROWS, OBJAVA as SIFRANT_OBJAVA,
                          DEJAVNOSTI as SIFRANT_DEJAVNOSTI, DEJAVNOSTI_BY_SIFRA as SIFRANT_DEJ_BY_SIFRA,
                          DEJAVNOST_CONTEXT as SIFRANT_DEJ_CONTEXT)
    from mtp import DEVICES as MTP_DEVICES, DATUM as MTP_DATUM
    from spa import SPA_ELIGIBILITY, STANDARD_TYPES
    from templates import TEMPLATES

    modules = {
        "contacts": {"loaded": bool(CONTACTS), "records": len(CONTACTS) if CONTACTS else 0},
        "drugs": {"loaded": True, "records": "live API (CBZ)"},
        "icd10": {"loaded": bool(ICD10_CODES), "records": len(ICD10_CODES) if ICD10_CODES else 0},
        "zzzs": {"loaded": bool(ZZZS_DRUGS), "records": f"{len(ZZZS_DRUGS)} drugs, {len(ZZZS_GROUPS)} groups, {len(ZZZS_RULES)} rules"},
        "egradiva": {"loaded": COLLECTION is not None, "records": COLLECTION.count() if COLLECTION else 0},
        "pravila": {"loaded": bool(PRAVILA_RULES), "records": f"{len(PRAVILA_RULES)} articles, dense={'yes' if PRAVILA_COLLECTION is not None else 'no'}"},
        "obracun": {"loaded": OBRACUN_COLLECTION is not None, "records": OBRACUN_COLLECTION.count() if OBRACUN_COLLECTION else 0},
        "sifranti": {"loaded": bool(SIFRANT_ROWS), "records": f"{len(SIFRANT_CAT)} šifrantov, {len(SIFRANT_ROWS)} code rows, objava {SIFRANT_OBJAVA.get('leto')}/{SIFRANT_OBJAVA.get('zap_st')}; dejavnost index: {len(SIFRANT_DEJAVNOSTI)} dejavnosti, {len(SIFRANT_DEJ_BY_SIFRA)} storitev mapiranih, {len(SIFRANT_DEJ_CONTEXT)} kontekstnih opomb"},
        "mtp": {"loaded": bool(MTP_DEVICES), "records": f"{len(MTP_DEVICES)} pripomočkov, stanje {MTP_DATUM}"},
        "spa": {"loaded": bool(SPA_ELIGIBILITY), "records": f"{len(SPA_ELIGIBILITY)} spas, {len(STANDARD_TYPES)} standards"},
        "templates": {"loaded": bool(TEMPLATES), "records": len(TEMPLATES) if TEMPLATES else 0},
    }

    for name, status in modules.items():
        init = _module_status.get(name, {})
        status["init_error"] = init.get("error")

    if _init_elapsed is None:
        status = "initializing"  # background startup (seed + model + collections) not done yet
    elif all(m["loaded"] for m in modules.values()):
        status = "ok"
    else:
        status = "degraded"
    last_refresh = None
    try:
        import refresh
        lr = (refresh.get_status() or {}).get("last_refresh") or {}
        if lr:
            eg = lr.get("egradiva", {})
            last_refresh = {"finished_at": lr.get("finished_at"), "status": lr.get("status"),
                            "sifranti_changed": lr.get("sifranti", {}).get("changed"),
                            "egradiva_added": len(eg.get("added_docs", [])),
                            "egradiva_chunks": eg.get("chunk_count")}
    except Exception:
        pass
    return {
        "status": status,
        "startup_time_s": _init_elapsed,
        "modules": modules,
        "last_refresh": last_refresh,
        "refresh_running": _refresh_running,
    }


@mcp.tool()
def preveri_stanje() -> dict:
    """Preveri stanje vseh modulov medicinskih orodij.

    Slovensko poimenovanje orodja health_check — deluje enako.
    """
    return health_check()


# ── List capabilities ────────────────────────────────────────────────

_SELF_TOOLS = {"list_capabilities", "seznam_zmoznosti"}


def _first_paragraph(text: str | None) -> str:
    """Return the first paragraph (up to the first blank line) of a docstring."""
    if not text:
        return ""
    lines: list[str] = []
    for line in text.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


@mcp.tool()
async def list_capabilities() -> dict:
    """List all available tools, resources, and prompts on this server.

    Returns a structured overview of every capability the server exposes,
    with a short description for each. Useful for answering "what can you do?"
    """
    tools = [
        {"name": t.name, "description": _first_paragraph(t.description)}
        for t in await mcp.list_tools()
        if t.name not in _SELF_TOOLS
    ]

    resources = [
        {"uri": str(r.uri), "name": r.name or "", "description": _first_paragraph(r.description)}
        for r in await mcp.list_resources()
    ]

    prompts = [
        {"name": p.name, "description": _first_paragraph(p.description)}
        for p in await mcp.list_prompts()
    ]

    return {"tools": tools, "resources": resources, "prompts": prompts}


@mcp.tool()
def seznam_zmoznosti() -> dict:
    """Izpiši vse razpoložljive zmožnosti (orodja, vire, pozive) strežnika.

    Slovensko poimenovanje orodja list_capabilities — deluje enako.
    """
    return list_capabilities()


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
def poisci_zdravilisce(query: str) -> dict:
    """Preveri katera zdravilišča v Sloveniji so upravičena do rehabilitacije
    iz obveznega zdravstvenega zavarovanja (ZZZS) glede na tip bolezni.

    Slovensko poimenovanje orodja get_spa_eligibility — deluje enako.
    """
    return _get_spa_eligibility(query)


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


@mcp.tool()
def poisci_smpc(query: str) -> dict:
    """Poišči SmPC (Povzetek glavnih značilnosti zdravila) za zdravilo.

    Slovensko poimenovanje orodja get_smpc — deluje enako.
    Prenese uradni SmPC PDF iz CBZ in izlušči klinično relevantne razdelke.
    """
    return _get_smpc(query)


if __name__ == "__main__":
    import os
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    # All heavy imports are done by now — start init in the background so it doesn't
    # contend with import/registration, then bring the server up immediately.
    _start_background_init()

    if transport in ("streamable-http", "sse"):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route, Mount

        async def _health(request):
            return JSONResponse({"status": "ok"})

        mcp_app = mcp.http_app(transport=transport)
        app = Starlette(
            routes=[
                Route("/health", _health),
                Mount("/", app=mcp_app),
            ],
            lifespan=mcp_app.lifespan,
        )

        import uvicorn
        uvicorn.run(app, host=host, port=port)
    else:
        mcp.run()
