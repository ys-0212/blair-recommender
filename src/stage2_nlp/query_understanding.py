"""Stage 2 query understanding — rule-based intent/constraint/expansion parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# maps common abbreviations to fuller vocabulary the embedding model handles better
DOMAIN_EXPANSIONS: dict[str, list[str]] = {
    "fps":      ["first person shooter", "shooter", "action"],
    "rpg":      ["role playing game", "role-playing", "adventure"],
    "mmo":      ["massively multiplayer online", "online", "multiplayer"],
    "mmorpg":   ["massively multiplayer online role playing game", "online rpg"],
    "rts":      ["real time strategy", "strategy"],
    "tbs":      ["turn based strategy", "strategy"],
    "jrpg":     ["japanese role playing game", "rpg", "adventure"],
    "moba":     ["multiplayer online battle arena", "competitive"],
    "soulslike":["dark souls style", "challenging", "action rpg"],
    "roguelike":["procedurally generated", "permadeath", "dungeon crawler"],
    "metroidvania": ["exploration", "platformer", "side-scroller"],
    "vr":       ["virtual reality", "immersive"],
    "co-op":    ["cooperative", "multiplayer", "together"],
    "coop":     ["cooperative", "multiplayer"],
    "indie":    ["independent game", "small studio"],
    "aaa":      ["triple a", "big budget", "blockbuster"],
    "arpg":     ["action role playing game", "action rpg"],
    "hack and slash": ["action", "combat", "beat em up"],
    "open world":     ["sandbox", "exploration", "non-linear"],
    "sandbox":        ["open world", "creative", "exploration"],
}

_PRICE_UNDER_RE = re.compile(
    r"(?:under|below|less\s+than|at\s+most|no\s+more\s+than|cheaper?\s+than)"
    r"\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PRICE_ABOVE_RE = re.compile(
    r"(?:over|above|more\s+than|at\s+least|greater\s+than)"
    r"\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PRICE_AROUND_RE = re.compile(
    r"(?:around|approximately|about|~)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Adjective pattern (simple heuristic — words before nouns that end in common adj suffixes)
_ADJ_RE = re.compile(
    r"\b(?:great|good|best|top|amazing|fun|exciting|challenging|easy|hard|"
    r"beautiful|gorgeous|stunning|dark|scary|funny|relaxing|competitive|"
    r"immersive|addictive|classic|new|old|cheap|expensive|popular|underrated|"
    r"hidden|rare|long|short|violent|family|single|multi|local|online|"
    r"free|paid|budget|premium|indie|aaa)\b",
    re.IGNORECASE,
)

# Common gaming nouns to anchor intent extraction
_NOUN_RE = re.compile(
    r"\b(?:game|games|title|titles|rpg|fps|shooter|adventure|action|strategy|"
    r"puzzle|platformer|horror|sports|racing|fighting|simulation|sandbox|"
    r"open.?world|story|multiplayer|singleplayer|co.?op|campaign|storyline|"
    r"character|characters|graphics|gameplay|controls|music|soundtrack|"
    r"console|pc|playstation|xbox|nintendo|switch)\b",
    re.IGNORECASE,
)


@dataclass
class QueryUnderstanding:
    """Structured result of parsing a user query."""
    raw_query:   str
    intent:      list[str] = field(default_factory=list)   # core topics
    attributes:  list[str] = field(default_factory=list)   # qualifiers
    constraints: dict[str, Any] = field(default_factory=dict)
    expanded:    list[str] = field(default_factory=list)   # domain-expanded terms
    final_query: str = ""  # enriched query text for embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query":   self.raw_query,
            "intent":      self.intent,
            "attributes":  self.attributes,
            "constraints": self.constraints,
            "expanded":    self.expanded,
            "final_query": self.final_query,
        }


def parse_query(query: str) -> QueryUnderstanding:
    """Rule-based parser: extracts intent, attributes, constraints, expansions."""
    q = query.strip()
    result = QueryUnderstanding(raw_query=q)

    # ---- 1. Extract constraints ----
    m = _PRICE_UNDER_RE.search(q)
    if m:
        result.constraints["max_price"] = float(m.group(1))

    m = _PRICE_ABOVE_RE.search(q)
    if m:
        result.constraints["min_price"] = float(m.group(1))

    m = _PRICE_AROUND_RE.search(q)
    if m:
        result.constraints["approx_price"] = float(m.group(1))

    # Bare "cheap" / "budget" → soft max price signal
    if re.search(r"\b(?:cheap|budget|affordable|free)\b", q, re.IGNORECASE):
        result.constraints.setdefault("price_preference", "low")
    if re.search(r"\b(?:premium|expensive|high.?end)\b", q, re.IGNORECASE):
        result.constraints.setdefault("price_preference", "high")

    # ---- 2. Extract attributes (adjectives) ----
    result.attributes = list({m.group().lower() for m in _ADJ_RE.finditer(q)})

    # ---- 3. Extract intent (gaming nouns / phrases) ----
    result.intent = list({m.group().lower() for m in _NOUN_RE.finditer(q)})

    # ---- 4. Domain expansion ----
    q_lower = q.lower()
    expanded_terms: list[str] = []
    for abbr, expansions in DOMAIN_EXPANSIONS.items():
        # Match as whole word or phrase
        pattern = r"\b" + re.escape(abbr) + r"\b"
        if re.search(pattern, q_lower):
            expanded_terms.extend(expansions)

    result.expanded = list(dict.fromkeys(expanded_terms))  # deduplicate, preserve order

    # ---- 5. Build enriched query for embedding ----
    parts = [q]
    if result.expanded:
        parts.append(" ".join(result.expanded))
    if result.attributes:
        parts.append(" ".join(result.attributes))
    result.final_query = " ".join(parts)

    return result


def parse_queries_batch(queries: list[str]) -> list[QueryUnderstanding]:
    """Parse a list of queries. Returns a list of QueryUnderstanding objects."""
    return [parse_query(q) for q in queries]
