"""Spa eligibility module - ZZZS mandatory health insurance rehabilitation."""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data (Priloga 19 Uredbe o programih storitev OZZ, avgust 2025)
# Source: https://api.zzzs.si/ZZZS/info/egradiva.nsf/0/14e583d0bfccfd93c12571860053dd2d/
# ---------------------------------------------------------------------------

SOURCE_URL = "https://api.zzzs.si/ZZZS/info/egradiva.nsf/0/14e583d0bfccfd93c12571860053dd2d/$FILE/zlo%C5%BEenka%20Seznam%20zdravili%C5%A1%C4%8D%20v%20Sloveniji_avgust%202025.pdf"

STANDARD_TYPES = {
    "tip 1": "Vnetne revmatske bolezni",
    "tip 2": "Degenerativni izvensklepni revmatizem",
    "tip 3": "Stanje po poškodbah in operacijah na lokomotornem sistemu s funkcijsko prizadetostjo",
    "tip 4": "Nevrološke bolezni, poškodbe in bolezni centralnega in perifernega živčnega sistema, vključno s cerebrovaskularnimi inzulti ter živčno-mišičnimi boleznimi",
    "tip 5": "Bolezni ter stanja po operacijah srca in ožilja",
    "tip 6": "Ginekološke bolezni, stanja po operativnih posegih v mali medenici, testisih in prsih",
    "tip 7": "Kožne bolezni",
    "tip 8": "Gastroenterološke in endokrine bolezni, stanja po operacijah",
    "tip 9": "Obolenja dihal",
}

SPA_ELIGIBILITY: List[Dict[str, Any]] = [
    {
        "name": "Terme Olimia",
        "url": "http://www.terme-olimia.com",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 7": "A"},
    },
    {
        "name": "Terme Čatež",
        "url": "http://www.terme-catez.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 4": "A", "tip 6": "B"},
    },
    {
        "name": "Mladinsko zdravilišče in letovišče RKS Debeli rtič",
        "url": "http://www.zdravilisce-debelirtic.org",
        "standards": {"tip 3": "A", "tip 7": "A", "tip 9": "A"},
    },
    {
        "name": "Terme Dobrna",
        "url": "http://www.terme-dobrna.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 4": "A", "tip 6": "A"},
    },
    {
        "name": "THERMANA – Zdravilišče Laško",
        "url": "http://www.thermana.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 4": "A", "tip 6": "B", "tip 7": "A"},
    },
    {
        "name": "Sava Turizem – Terme 3000",
        "url": "http://www.sava-hotels-resorts.com",
        "standards": {"tip 1": "B", "tip 2": "B", "tip 3": "B", "tip 7": "B"},
    },
    {
        "name": "Sava Turizem – Terme Ptuj",
        "url": "http://www.sava-hotels-resorts.com",
        "standards": {"tip 2": "B", "tip 3": "B"},
    },
    {
        "name": "Sava Turizem – Zdravilišče Radenci",
        "url": "http://www.sava-hotels-resorts.com",
        "standards": {"tip 2": "A", "tip 4": "A", "tip 5": "A"},
    },
    {
        "name": "Istrabenz Turizem – Terme Portorož",
        "url": "http://www.lifeclass.net",
        "standards": {"tip 1": "B", "tip 2": "B", "tip 3": "B", "tip 7": "B"},
    },
    {
        "name": "Zdravilišče Rogaška Slatina",
        "url": "http://www.rogaska-medical.com",
        "standards": {"tip 8": "A"},
    },
    {
        "name": "Terme Krka – Zdravilišče Dolenjske toplice",
        "url": "http://www.terme-krka.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 4": "A", "tip 6": "B"},
    },
    {
        "name": "Terme Krka – Talaso Strunjan",
        "url": "http://www.terme-krka.si",
        "standards": {"tip 2": "B", "tip 3": "B", "tip 7": "B", "tip 9": "B"},
    },
    {
        "name": "Terme Krka – Zdravilišče Šmarješke toplice",
        "url": "http://www.terme-krka.si",
        "standards": {"tip 2": "A", "tip 3": "A", "tip 4": "A", "tip 5": "A"},
    },
    {
        "name": "Terme Topolšica",
        "url": "http://www.t-topolsica.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 4": "A"},
    },
    {
        "name": "Terme resort (Rimske terme)",
        "url": "http://www.rimske-terme.si",
        "standards": {"tip 1": "B", "tip 2": "B", "tip 3": "B", "tip 4": "B", "tip 5": "B", "tip 6": "B", "tip 7": "B", "tip 9": "B"},
    },
    {
        "name": "Unior d.d. – Terme Zreče",
        "url": "http://www.terme-zrece.si",
        "standards": {"tip 1": "A", "tip 2": "A", "tip 3": "A", "tip 6": "A", "tip 9": "A"},
    },
    {
        "name": "MC Medicor",
        "url": "http://www.mcmedicor.si",
        "standards": {"tip 5": "B"},
    },
]


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _score_text(text: str, query: str) -> int:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0
    text_tokens = set(_tokenize(text))
    return len(query_tokens & text_tokens)


def initialize_spa() -> None:
    """Initialize spa module (data is embedded, nothing to load)."""
    print(f"Spa module ready: {len(SPA_ELIGIBILITY)} spas, {len(STANDARD_TYPES)} standards")


def _get_spa_eligibility(query: str) -> Dict[str, Any]:
    """Check spa eligibility for rehabilitation under ZZZS mandatory insurance."""
    q = query.strip().lower()
    if not q:
        return {
            "found": False,
            "error": "Query must not be empty",
            "standards": {k: v for k, v in STANDARD_TYPES.items()},
            "source_url": SOURCE_URL,
        }

    # Check for direct standard type match (e.g. "tip 1", "tip 5")
    matched_standards: List[str] = []
    for tip_key in STANDARD_TYPES:
        if tip_key in q:
            matched_standards.append(tip_key)

    # Score standard descriptions against query — keep only the best-scoring tier
    if not matched_standards:
        scored_standards = [
            (_score_text(desc, q), key)
            for key, desc in STANDARD_TYPES.items()
        ]
        scored_standards.sort(key=lambda x: x[0], reverse=True)
        if scored_standards and scored_standards[0][0] > 0:
            best = scored_standards[0][0]
            matched_standards = [key for score, key in scored_standards if score == best]

    # Score spa names against query
    scored_spas = [
        (_score_text(spa["name"], q), spa)
        for spa in SPA_ELIGIBILITY
    ]
    scored_spas.sort(key=lambda x: x[0], reverse=True)
    best_spa_score = scored_spas[0][0] if scored_spas else 0

    results: List[Dict[str, Any]] = []

    # If query matches a spa name, show that spa's profile
    if best_spa_score > 0 and (
        not matched_standards
        or best_spa_score >= max(
            _score_text(STANDARD_TYPES[s], q) for s in matched_standards
        )
    ):
        matched_spas = [spa for score, spa in scored_spas if score > 0]
        for spa in matched_spas:
            results.append({
                "spa": spa["name"],
                "url": spa["url"],
                "standards": {
                    tip_key: {
                        "level": level,
                        "label": "Primarna dejavnost" if level == "A" else "Sekundarna dejavnost",
                        "description": STANDARD_TYPES.get(tip_key, tip_key),
                    }
                    for tip_key, level in sorted(spa["standards"].items())
                },
            })

    # Show spas matching the standard type(s)
    standard_results: List[Dict[str, Any]] = []
    if matched_standards:
        for tip_key in matched_standards:
            desc = STANDARD_TYPES[tip_key]
            primary = []
            secondary = []
            for spa in SPA_ELIGIBILITY:
                level = spa["standards"].get(tip_key)
                if level == "A":
                    primary.append({"name": spa["name"], "url": spa["url"]})
                elif level == "B":
                    secondary.append({"name": spa["name"], "url": spa["url"]})
            standard_results.append({
                "standard": tip_key,
                "description": desc,
                "primary_A": primary,
                "secondary_B": secondary,
            })

    if not results and not standard_results:
        return {
            "found": False,
            "query": query,
            "error": "No matches found",
            "available_standards": {k: v for k, v in STANDARD_TYPES.items()},
            "source_url": SOURCE_URL,
        }

    return {
        "found": True,
        "query": query,
        "spa_profiles": results if results else None,
        "standard_results": standard_results if standard_results else None,
        "legend": {
            "A": "Primarna dejavnost – polna rehabilitacija z negovalnim oddelkom",
            "B": "Sekundarna dejavnost – rehabilitacija brez negovalnega oddelka",
        },
        "source_url": SOURCE_URL,
    }
