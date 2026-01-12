import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

LOCALISATION_DIR = Path(__file__).resolve().parent / "localisation"
_LOGGER = logging.getLogger(__name__)


def normalize_language(lang: str) -> str:
    cleaned = (lang or "").strip().lower().replace("-", "_")
    return cleaned


def language_base(lang: str) -> str:
    cleaned = normalize_language(lang)
    if not cleaned:
        return ""
    return cleaned.split("_", 1)[0]


def _language_candidates(lang: str) -> Iterable[str]:
    cleaned = normalize_language(lang)
    if not cleaned:
        return ["en"]
    base = language_base(cleaned)
    candidates = [cleaned]
    if base and base != cleaned:
        candidates.append(base)
    if "en" not in candidates:
        candidates.append("en")
    return candidates


def _load_locale_file(language: str) -> Dict[str, Any]:
    path = LOCALISATION_DIR / language / "strings.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _merged_localisation(lang: str) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {
        "strings": {},
        "gear_type_labels": {},
        "relation_labels": {},
    }
    for code in reversed(list(_language_candidates(lang))):
        data = _load_locale_file(code)
        for key in merged:
            merged[key].update(data.get(key, {}))
    _warn_missing_strings(lang)
    return merged


def tr(lang: str, key: str) -> str:
    return _merged_localisation(lang)["strings"].get(key, key)


def gear_type_label(gear_type: str, lang: str) -> str:
    return _merged_localisation(lang)["gear_type_labels"].get(gear_type, gear_type)


def relation_label(relation: str, lang: str) -> str:
    return _merged_localisation(lang)["relation_labels"].get(relation, relation)


def available_languages() -> List[str]:
    if not LOCALISATION_DIR.exists():
        return ["en"]
    codes = []
    for entry in LOCALISATION_DIR.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "strings.json").exists():
            codes.append(entry.name)
    return sorted(codes) or ["en"]


def resolve_language(lang: str) -> str:
    for code in _language_candidates(lang):
        if (LOCALISATION_DIR / code / "strings.json").exists():
            return code
    return "en"


def language_display_name(lang: str) -> str:
    normalized = normalize_language(lang)
    data = _load_locale_file(normalized)
    name = data.get("strings", {}).get("language_name")
    if name:
        return name
    base = language_base(normalized)
    if base and base != normalized:
        base_name = _load_locale_file(base).get("strings", {}).get("language_name")
        if base_name:
            return base_name
    return normalized or "en"


def resolve_readme_path(lang: str) -> Path:
    normalized = normalize_language(lang)
    base = language_base(normalized)
    candidates = []
    for code in (normalized, base):
        if code:
            candidates.append(LOCALISATION_DIR / code / "README.md")
    for code in (normalized, base):
        if code:
            candidates.append(LOCALISATION_DIR.parent / f"README.{code}.md")
    candidates.append(LOCALISATION_DIR.parent / "README.md")
    for path in candidates:
        if path.exists():
            return path
    return LOCALISATION_DIR.parent / "README.md"


@lru_cache(maxsize=None)
def _missing_string_keys(lang: str) -> List[str]:
    normalized = normalize_language(lang)
    if not normalized:
        return []
    if normalized == "en":
        return []
    en_strings = _load_locale_file("en").get("strings", {})
    if not en_strings:
        return []
    locale_strings = _load_locale_file(normalized).get("strings", {})
    if not locale_strings:
        return []
    return sorted(set(en_strings.keys()) - set(locale_strings.keys()))


def _warn_missing_strings(lang: str) -> None:
    missing = _missing_string_keys(lang)
    if missing:
        _LOGGER.warning(
            "Missing localisation strings for %s: %s",
            normalize_language(lang),
            ", ".join(missing),
        )
