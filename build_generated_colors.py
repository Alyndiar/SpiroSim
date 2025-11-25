"""
build_generated_colors.py

Builds generated_colors.py containing a large COLOR_NAME_TO_HEX dict.

Sources:

1. Matplotlib named colors:
   - BASE_COLORS
   - CSS4_COLORS
   - TABLEAU_COLORS
   - XKCD_COLORS

2. X11 color names from:
   https://pdos.csail.mit.edu/~jinyang/rgb.html
   (HTML rendering of rgb.txt)

3. Wikipedia "List of colors: A–F", "G–M", "N–Z" only.

Priority for name clashes (first definition wins, AFTER normalization):
    Matplotlib: BASE -> CSS4 -> TABLEAU -> XKCD
    then X11
    then Wikipedia A–Z.

Name normalization:
    - lowercase
    - remove ALL whitespace characters (spaces, tabs, etc.)

Example:
    "Light Goldenrod 3" -> "lightgoldenrod3"

Run:

    python build_generated_colors.py

This will create/overwrite generated_colors.py in the same directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import pandas as pd
from matplotlib import colors as mcolors
from bs4 import BeautifulSoup


# ======================================================================
# Normalization & hex helpers
# ======================================================================

HEX_RE = re.compile(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


def normalize_name_for_key(name: str) -> str:
    """
    Normalize a color name for keys:
      - strip leading/trailing whitespace
      - lowercase
      - remove ALL internal whitespace (spaces, tabs, etc.)
    """
    return re.sub(r"\s+", "", name.strip().lower())


def normalize_hex(hex_value: str) -> str:
    """
    Normalize any valid hex to #RRGGBB uppercase.
    Accepts #rgb or #rrggbb.
    """
    hex_value = hex_value.strip()
    if not hex_value.startswith("#"):
        raise ValueError(f"Not a hex color: {hex_value!r}")

    if len(hex_value) == 4:  # #rgb
        h = hex_value[1:]
        hex_value = "#" + "".join(c * 2 for c in h)

    if len(hex_value) != 7 or not HEX_RE.match(hex_value):
        raise ValueError(f"Invalid hex color: {hex_value!r}")

    return hex_value.upper()


# ======================================================================
# 1. Matplotlib named colors
# ======================================================================

def build_matplotlib_color_map() -> Dict[str, str]:
    """
    Build a color dict from Matplotlib named colors in the desired order:
        BASE_COLORS -> CSS4_COLORS -> TABLEAU_COLORS -> XKCD_COLORS

    First definition for a (normalized) name wins.
    """
    name_to_hex: Dict[str, str] = {}

    def add_source(d: Dict[str, str | tuple | list]):
        nonlocal name_to_hex
        for name, value in d.items():
            # For XKCD colors, also provide a version without "xkcd:" prefix
            if isinstance(name, str) and name.startswith("xkcd:"):
                plain = name[5:]
                candidates = [plain, name]
            else:
                candidates = [name]

            for cand in candidates:
                key = normalize_name_for_key(str(cand))
                if key in name_to_hex:
                    continue
                hex_value = mcolors.to_hex(value, keep_alpha=False)
                name_to_hex[key] = normalize_hex(hex_value)

    # Priority: BASE -> CSS4 -> TABLEAU -> XKCD
    add_source(mcolors.BASE_COLORS)
    add_source(mcolors.CSS4_COLORS)
    add_source(mcolors.TABLEAU_COLORS)
    add_source(mcolors.XKCD_COLORS)

    return name_to_hex


# ======================================================================
# 2. X11 color names from MIT rgb.html (BeautifulSoup)
# ======================================================================

X11_RGB_URL = "https://pdos.csail.mit.edu/~jinyang/rgb.html"


def fetch_x11_colors() -> Dict[str, str]:
    """
    Fetch X11 colors from the HTML rgb.txt rendering at:

        https://pdos.csail.mit.edu/~jinyang/rgb.html

    The page is essentially rgb.txt rendered as text with the pattern:

        name
        #RRGGBB
        RGB=(r,g,b) [next name]
        #RRGGBB
        RGB=(...)

    or:

        ... RGB=(...) [next name]
        #RRGGBB
        RGB=(...)

    So:
        - Some names appear on their own line (e.g. "snow", "snow1").
        - Some names appear AFTER an RGB line on the same line
          (e.g. "RGB=(255,250,250) ghost white").

    We:
        - Parse the HTML with BeautifulSoup.
        - Walk the text line-by-line.
        - Track a pending name; when we hit a "#RRGGBB" line, we bind it
          to the pending name (if any).
        - Extract names both from standalone lines and from the tail of
          "RGB=(...) name" lines.

    All names are normalized by normalize_name_for_key.

    Returns dict: normalized_name -> #RRGGBB
    """
    resp = requests.get(X11_RGB_URL, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n")
    lines = text.splitlines()

    x11: Dict[str, str] = {}
    pending_name: str | None = None

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Hex line => if we have a pending name, bind it
        if s.startswith("#"):
            if pending_name:
                key = normalize_name_for_key(pending_name)
                try:
                    hex_norm = normalize_hex(s)
                except ValueError:
                    pending_name = None
                    continue
                # First definition wins for this layer
                x11.setdefault(key, hex_norm)
                pending_name = None
            continue

        # RGB line, may contain the next name at the tail:
        #   "RGB=(255,250,250) ghost white"
        if "RGB=" in s:
            idx = s.find(")")
            if idx != -1:
                tail = s[idx + 1 :].strip()
                if tail:
                    pending_name = tail
            continue

        # Otherwise, treat it as a standalone name line:
        #   "snow", "snow1", "AntiqueWhite1", "gray0", "grey0", etc.
        pending_name = s

    return x11


# ======================================================================
# 3. Wikipedia "List of colors: A–F, G–M, N–Z" (BeautifulSoup + pandas)
# ======================================================================

WIKI_COLOR_URLS = [
    "https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F",
    "https://en.wikipedia.org/wiki/List_of_colors:_G%E2%80%93M",
    "https://en.wikipedia.org/wiki/List_of_colors:_N%E2%80%93Z",
]


def _wiki_headers() -> Dict[str, str]:
    """
    Custom headers to avoid 403 and identify the scraper politely.
    Update the contact info if you want.
    """
    return {
        "User-Agent": (
            "DanyColorScraper/1.0 "
            "(contact: you@example.com; purpose: build local color dict)"
        ),
        "Accept-Language": "en-US,en;q=0.8",
    }


def fetch_wikipedia_color_table(url: str) -> List[Tuple[str, str]]:
    """
    Fetch (name, hex) pairs from one Wikipedia 'List of colors' page.

    We:
        - GET the page with a custom User-Agent,
        - parse with BeautifulSoup,
        - find all 'wikitable' tables,
        - let pandas.read_html() parse each table individually,
        - look for a 'Hex' and 'Name' column in each.

    Returns list[(name, hex)].
    """
    resp = requests.get(url, headers=_wiki_headers(), timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables_html = soup.find_all("table", class_="wikitable")

    results: List[Tuple[str, str]] = []

    for tbl in tables_html:
        # Convert just this table to HTML string for pandas
        html_str = str(tbl)
        try:
            dfs = pd.read_html(html_str)
        except ValueError:
            continue
        if not dfs:
            continue
        df = dfs[0]

        hex_cols = [c for c in df.columns if isinstance(c, str) and "hex" in c.lower()]
        name_cols = [c for c in df.columns if isinstance(c, str) and "name" in c.lower()]
        if not hex_cols or not name_cols:
            continue

        hex_col = hex_cols[0]
        name_col = name_cols[0]

        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            hex_val = str(row[hex_col]).strip()
            if not name or not isinstance(hex_val, str):
                continue
            if not hex_val.startswith("#"):
                continue
            try:
                hex_norm = normalize_hex(hex_val)
            except ValueError:
                continue
            results.append((name, hex_norm))

    return results


def fetch_wikipedia_colors() -> Dict[str, str]:
    """
    Fetch all A–Z Wikipedia color names from the three list pages.

    Only those three pages; we DO NOT scrape other lists
    like Crayola, RAL, etc.

    Returns dict: normalized_name -> #RRGGBB

    If some pages fail (HTTP error), they are skipped with a warning.
    """
    name_to_hex: Dict[str, str] = {}

    for url in WIKI_COLOR_URLS:
        try:
            entries = fetch_wikipedia_color_table(url)
        except requests.HTTPError as e:
            print(f"[WARNING] Could not fetch {url} ({e}). Skipping this page.")
            continue

        for name, hex_value in entries:
            key = normalize_name_for_key(name)
            # Inside Wikipedia layer, first definition wins
            name_to_hex.setdefault(key, hex_value)

    return name_to_hex


# ======================================================================
# 4. Build final mapping with priority and write generated_colors.py
# ======================================================================

def build_master_mapping() -> Dict[str, str]:
    """
    Build the final name -> hex mapping with the requested priority:

        1. Matplotlib (Base -> CSS4 -> Tableau -> XKCD)
        2. X11
        3. Wikipedia A–Z

    First definition wins *per normalized name*.
    Different names mapping to the same hex are all preserved (because
    we don't deduplicate by value, only by name).
    """
    master: Dict[str, str] = {}

    def add_layer(layer: Dict[str, str], label: str):
        print(f"  Merging layer: {label} ({len(layer)} entries)")
        for key, hex_val in layer.items():
            key_norm = normalize_name_for_key(key)
            hex_norm = normalize_hex(hex_val)
            if key_norm in master:
                continue
            master[key_norm] = hex_norm

    # 1. Matplotlib
    print("Building Matplotlib colors...")
    mpl_map = build_matplotlib_color_map()
    add_layer(mpl_map, "Matplotlib")

    # 2. X11
    print("Fetching X11 colors...")
    x11_map = fetch_x11_colors()
    add_layer(x11_map, "X11")

    # 3. Wikipedia
    print("Fetching Wikipedia A–Z colors...")
    wiki_map = fetch_wikipedia_colors()
    add_layer(wiki_map, "Wikipedia A–Z")

    print(f"Total unique normalized names in master mapping: {len(master)}")
    return master


def write_generated_py(mapping: Dict[str, str], out_path: Path):
    """
    Write generated_colors.py with a single COLOR_NAME_TO_HEX dict,
    containing normalized_name -> #RRGGBB.
    """
    items = sorted(mapping.items(), key=lambda kv: kv[0])

    header = '''"""
Auto-generated color name dictionary.

COLOR_NAME_TO_HEX:
    - keys are normalized color names:
        * lowercase
        * all whitespace removed
    - values are #RRGGBB hex strings

Sources (via build_generated_colors.py):
    - Matplotlib named colors (Base, CSS4, Tableau, XKCD)
    - X11 color names from pdos.csail.mit.edu rgb.html
    - Wikipedia "List of colors: A–F, G–M, N–Z" (CC BY-SA 4.0)

Do NOT edit this file by hand.
Regenerate it by running build_generated_colors.py.
"""

COLOR_NAME_TO_HEX = {
'''

    lines = [header]

    for name, hex_value in items:
        # name already normalized (lowercase, no whitespace)
        lines.append(f"    {name!r}: {hex_value!r},\n")

    lines.append("}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    master = build_master_mapping()
    out_path = Path(__file__).with_name("generated_colors.py")
    write_generated_py(master, out_path)
    print(f"Wrote {len(master)} entries to {out_path}")


if __name__ == "__main__":
    main()
