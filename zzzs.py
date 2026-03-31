"""ZZZS limitations & prescribing rules module - CBZ CSV data + bundled rules."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cache import is_cache_fresh, read_cache, write_cache
from fastmcp.tools.tool import ToolResult

# Configuration
DATA_DIR = Path(__file__).parent / "data"
RULES_FILE = DATA_DIR / "zzzs_rules.json"
CACHE_NAME_DRUGS = "zzzs_drugs"
CACHE_NAME_GROUPS = "zzzs_groups"
CACHE_TTL_HOURS = 24

CBZ_CSV_URLS = {
    "sif30": "https://www.cbz.si/cbz2/sif30.csv",
    "siftsz": "https://www.cbz.si/cbz2/siftsz.csv",
    "siftszmzz": "https://www.cbz.si/cbz2/siftszmzz.csv",
}

# Module-level state
ZZZS_DRUGS: List[Dict[str, Any]] = []
ZZZS_GROUPS: List[Dict[str, Any]] = []
ZZZS_RULES: List[Dict[str, Any]] = []
DRUG_VECTORIZER: Optional[TfidfVectorizer] = None
DRUG_VECTORS: Optional[Any] = None
RULES_VECTORIZER: Optional[TfidfVectorizer] = None
RULES_VECTORS: Optional[Any] = None


def _fetch_csv(url: str, encoding: str = "windows-1250") -> List[Dict[str, str]]:
    """Download and parse a CSV file from CBZ."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.content.decode(encoding, errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=";")
        return [dict(row) for row in reader]
    except Exception as e:
        print(f"Failed to fetch CSV from {url}: {e}")
        return []


def _fetch_zzzs_drug_list() -> List[Dict[str, Any]]:
    """Download and parse sif30.csv - main drug list with limitations."""
    rows = _fetch_csv(CBZ_CSV_URLS["sif30"])
    drugs = []
    for row in rows:
        drug = {
            "code": row.get("Nacionalna šifra zdravila", "").strip(),
            "name": row.get("Ime zdravila", "").strip(),
            "atc": row.get("ATC", "").strip(),
            "atc_name": row.get("Naziv ATC", "").strip(),
            "list": row.get("Lista", "").strip(),
            "limitation": row.get("Omejitev predpisovanja in izdajanja v breme javnih sredstev", "").strip(),
            "valid_from": row.get("Datum pričetka veljavnosti", "").strip(),
        }
        if drug["name"]:
            drugs.append(drug)
    return drugs


def _fetch_zzzs_therapeutic_groups() -> List[Dict[str, Any]]:
    """Download and parse siftsz.csv - therapeutic groups with NPV pricing."""
    rows = _fetch_csv(CBZ_CSV_URLS["siftsz"])
    groups = []
    for row in rows:
        group = {
            "group_name": row.get("Terapevtska skupina zdravil", "").strip(),
            "group_class": row.get("Razred terapevtske skupine zdravil", "").strip(),
            "code": row.get("Nacionalna šifra zdravila", "").strip(),
            "drug_name": row.get("Ime zdravila", "").strip(),
            "list": row.get("Lista", "").strip(),
            "regulated_price": row.get("Regulirana cena", "").strip(),
            "agreed_price": row.get("Dogovorjena cena", "").strip(),
            "npv": row.get("Najvišja priznana vrednost", "").strip(),
            "surcharge": row.get("Informativno doplačilo z DDV", "").strip(),
            "on_mzz_list": row.get("Zdravilo je na seznamu MZZ z NPV", "").strip(),
        }
        if group["group_name"] or group["drug_name"]:
            groups.append(group)
    return groups


def _load_rules() -> List[Dict[str, Any]]:
    """Load bundled ZZZS rules from data/zzzs_rules.json."""
    if RULES_FILE.exists():
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ZZZS rules: {e}")
    return []


def _build_drug_index(drugs: List[Dict[str, Any]]) -> None:
    """Build TF-IDF index on drug name + ATC + limitation text."""
    global DRUG_VECTORIZER, DRUG_VECTORS
    if not drugs:
        return
    corpus = [
        f"{d['name']} {d.get('atc', '')} {d.get('limitation', '')}"
        for d in drugs
    ]
    DRUG_VECTORIZER = TfidfVectorizer()
    DRUG_VECTORIZER.fit(corpus)
    DRUG_VECTORS = DRUG_VECTORIZER.transform(corpus)


def _build_rules_index(rules: List[Dict[str, Any]]) -> None:
    """Build TF-IDF index on rules text."""
    global RULES_VECTORIZER, RULES_VECTORS
    if not rules:
        return
    corpus = [
        f"{r.get('title', '')} {r.get('section', '')} {r.get('text', '')} {r.get('category', '')}"
        for r in rules
    ]
    RULES_VECTORIZER = TfidfVectorizer()
    RULES_VECTORIZER.fit(corpus)
    RULES_VECTORS = RULES_VECTORIZER.transform(corpus)


def initialize_zzzs() -> None:
    """Initialize ZZZS module: download CSVs, load rules, build indices."""
    global ZZZS_DRUGS, ZZZS_GROUPS, ZZZS_RULES

    print("Loading ZZZS data...")

    # Load drug list
    if is_cache_fresh(CACHE_NAME_DRUGS, max_age_hours=CACHE_TTL_HOURS):
        cached = read_cache(CACHE_NAME_DRUGS)
        if cached:
            ZZZS_DRUGS = cached
            print(f"Loaded {len(ZZZS_DRUGS)} ZZZS drugs from cache")
        else:
            _refresh_drug_data()
    else:
        _refresh_drug_data()

    # Load therapeutic groups
    if is_cache_fresh(CACHE_NAME_GROUPS, max_age_hours=CACHE_TTL_HOURS):
        cached = read_cache(CACHE_NAME_GROUPS)
        if cached:
            ZZZS_GROUPS = cached
            print(f"Loaded {len(ZZZS_GROUPS)} ZZZS therapeutic groups from cache")
        else:
            _refresh_group_data()
    else:
        _refresh_group_data()

    # Build drug index
    _build_drug_index(ZZZS_DRUGS)

    # Load bundled rules
    ZZZS_RULES = _load_rules()
    if ZZZS_RULES:
        _build_rules_index(ZZZS_RULES)
        print(f"Loaded {len(ZZZS_RULES)} ZZZS rules")
    else:
        print("No bundled ZZZS rules available (data/zzzs_rules.json not found)")

    print("ZZZS module initialized")


def _refresh_drug_data() -> None:
    """Fetch fresh drug data from CBZ CSV."""
    global ZZZS_DRUGS
    try:
        drugs = _fetch_zzzs_drug_list()
        if drugs:
            ZZZS_DRUGS = drugs
            write_cache(CACHE_NAME_DRUGS, drugs)
            print(f"Fetched {len(drugs)} drugs from CBZ CSV")
            return
    except Exception as e:
        print(f"Warning: Failed to fetch ZZZS drug list: {e}")

    # Fall back to stale cache
    cached = read_cache(CACHE_NAME_DRUGS)
    if cached:
        ZZZS_DRUGS = cached
        print(f"Loaded {len(ZZZS_DRUGS)} ZZZS drugs from stale cache")


def _refresh_group_data() -> None:
    """Fetch fresh therapeutic group data from CBZ CSV."""
    global ZZZS_GROUPS
    try:
        groups = _fetch_zzzs_therapeutic_groups()
        if groups:
            ZZZS_GROUPS = groups
            write_cache(CACHE_NAME_GROUPS, groups)
            print(f"Fetched {len(groups)} therapeutic groups from CBZ CSV")
            return
    except Exception as e:
        print(f"Warning: Failed to fetch ZZZS therapeutic groups: {e}")

    cached = read_cache(CACHE_NAME_GROUPS)
    if cached:
        ZZZS_GROUPS = cached
        print(f"Loaded {len(ZZZS_GROUPS)} ZZZS groups from stale cache")


# ── Formatting helpers ──────────────────────────────────────────────

def _blockquote(text: str) -> str:
    if not text:
        return ""
    return "\n".join(f"> {line}" for line in text.strip().splitlines())


def _format_zzzs_limitation_md(data: dict) -> str:
    if not data.get("found"):
        return f"No ZZZS drug limitations found for \"{data.get('query', '')}\". {data.get('error', '')}"

    lines = [f"**ZZZS omejitve** | {data['count']} results for \"{data['query']}\"", ""]

    for r in data["results"]:
        lines.append("---")
        lines.append("")
        name = r.get("name", "")
        atc = r.get("atc", "")
        lista = r.get("list", "")
        conf = r.get("confidence", "")
        lines.append(f"**{name}** ({atc}) | Lista: {lista} | Confidence: {conf}")
        lines.append("")

        limitation = r.get("limitation", "")
        if limitation:
            lines.append(_blockquote(limitation))
        else:
            lines.append("Brez omejitve.")
        lines.append("")

        valid = r.get("valid_from", "")
        if valid:
            lines.append(f"Veljavno od: {valid}")
            lines.append("")

    return "\n".join(lines)


def _format_rules_md(data: dict, heading: str = "Pravila OZZ") -> str:
    if not data.get("found"):
        q = data.get("query", data.get("category", ""))
        return f"No ZZZS rules found for \"{q}\". {data.get('error', '')}"

    q = data.get("query", data.get("category", ""))
    lines = [f"**{heading}** | {data['count']} results for \"{q}\"", ""]

    for r in data["results"]:
        lines.append("---")
        lines.append("")
        title = r.get("title", "Untitled")
        art_num = r.get("article_number")
        if art_num:
            lines.append(f"### Čl. {art_num}: {title}")
        else:
            lines.append(f"### {title}")
        lines.append("")

        meta_parts = []
        if r.get("section"):
            meta_parts.append(f"**Section**: {r['section']}")
        if r.get("category"):
            meta_parts.append(f"**Category**: {r['category']}")
        if r.get("confidence") is not None:
            meta_parts.append(f"**Confidence**: {r['confidence']}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))
            lines.append("")

        if r.get("text"):
            lines.append(_blockquote(r["text"]))
            lines.append("")

        source_url = r.get("source_url", "")
        if source_url:
            lines.append(f"[Vir]({source_url})")
            lines.append("")

    return "\n".join(lines)


def _wrap(data: dict, formatter) -> ToolResult:
    return ToolResult(content=formatter(data), structured_content=data)


# ── Tool functions ──────────────────────────────────────────────────

def _get_zzzs_limitation(query: str) -> ToolResult:
    """Search ZZZS drug limitations by drug name, ATC, or limitation text."""
    if not ZZZS_DRUGS:
        _refresh_drug_data()
        _build_drug_index(ZZZS_DRUGS)
        if not ZZZS_DRUGS:
            err = {"found": False, "error": "No ZZZS drug data available"}
            return _wrap(err, _format_zzzs_limitation_md)

    if DRUG_VECTORIZER is None or DRUG_VECTORS is None:
        err = {"found": False, "error": "Drug search index not initialized"}
        return _wrap(err, _format_zzzs_limitation_md)

    query = query.strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return _wrap(err, _format_zzzs_limitation_md)

    query_vector = DRUG_VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vector, DRUG_VECTORS).flatten()

    results = []
    for idx, score in enumerate(similarities):
        if score > 0.2:
            drug = ZZZS_DRUGS[idx]
            results.append({
                **drug,
                "confidence": round(float(score), 4),
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if not results:
        err = {"found": False, "error": "No matching drugs found", "query": query}
        return _wrap(err, _format_zzzs_limitation_md)

    data = {
        "found": True,
        "count": len(results[:20]),
        "query": query,
        "results": results[:20],
    }
    return _wrap(data, _format_zzzs_limitation_md)


def _get_zzzs_prescribing_rules(topic: str) -> ToolResult:
    """Search bundled ZZZS rules about prescribing, referrals, coverage."""
    if not ZZZS_RULES:
        err = {"found": False, "error": "No ZZZS rules data available (data/zzzs_rules.json not found)"}
        return _wrap(err, _format_rules_md)

    if RULES_VECTORIZER is None or RULES_VECTORS is None:
        err = {"found": False, "error": "Rules search index not initialized"}
        return _wrap(err, _format_rules_md)

    topic = topic.strip()
    if not topic:
        err = {"found": False, "error": "Topic must not be empty"}
        return _wrap(err, _format_rules_md)

    query_vector = RULES_VECTORIZER.transform([topic])
    similarities = cosine_similarity(query_vector, RULES_VECTORS).flatten()

    results = []
    for idx, score in enumerate(similarities):
        if score > 0.15:
            rule = ZZZS_RULES[idx]
            result = {**rule, "confidence": round(float(score), 4)}
            result.setdefault("source_url", "")
            results.append(result)

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if not results:
        err = {"found": False, "error": "No matching rules found", "query": topic}
        return _wrap(err, _format_rules_md)

    data = {
        "found": True,
        "count": len(results[:10]),
        "query": topic,
        "results": results[:10],
    }
    return _wrap(data, _format_rules_md)


def _browse_zzzs_rules(category: str) -> ToolResult:
    """Browse ZZZS rules by category/section. Falls back to TF-IDF search."""
    if not ZZZS_RULES:
        err = {"found": False, "error": "No ZZZS rules data available"}
        return _wrap(err, _format_rules_md)

    category = category.strip().lower()
    if not category:
        err = {"found": False, "error": "Category must not be empty"}
        return _wrap(err, _format_rules_md)

    # Exact category match
    matches = [r for r in ZZZS_RULES if r.get("category", "").lower() == category]

    # Partial match on category or section
    if not matches:
        matches = [
            r for r in ZZZS_RULES
            if category in r.get("category", "").lower()
            or category in r.get("section", "").lower()
            or category in r.get("chapter", "").lower()
        ]

    # Fall back to TF-IDF search (already returns ToolResult)
    if not matches:
        return _get_zzzs_prescribing_rules(category)

    results = [
        {"title": r["title"], "section": r.get("section", ""), "category": r.get("category", ""),
         "article_number": r.get("article_number", 0), "text": r["text"],
         "source_url": r.get("source_url", "")}
        for r in matches
    ]
    results.sort(key=lambda x: x.get("article_number", 0))

    data = {
        "found": True,
        "count": len(results),
        "category": category,
        "results": results,
    }
    return _wrap(data, _format_rules_md)


def _list_zzzs_categories() -> Dict[str, Any]:
    """List all distinct categories and their article counts."""
    if not ZZZS_RULES:
        return {"found": False, "error": "No ZZZS rules data available"}

    counts: Dict[str, int] = {}
    for r in ZZZS_RULES:
        cat = r.get("category", "drugo")
        counts[cat] = counts.get(cat, 0) + 1

    categories = [
        {"category": cat, "article_count": count}
        for cat, count in sorted(counts.items(), key=lambda x: -x[1])
    ]

    return {
        "found": True,
        "total_rules": len(ZZZS_RULES),
        "total_categories": len(categories),
        "categories": categories,
    }


if __name__ == "__main__":
    print("Testing ZZZS module...")
    initialize_zzzs()

    print("\n--- Drug limitation search: amoksicilin ---")
    result = _get_zzzs_limitation("amoksicilin")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
