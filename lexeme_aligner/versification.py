"""Versification remap — bring a target Bible's verses onto the SOURCE (spine) numbering so verse-by-verse
matching lines up.

A target not numbered like the spine (e.g. Russian Synodal = orthodox / LXX tradition) has its verses
shifted against the Hebrew/Greek spine, so alignment silently breaks (rus contested-oracle stuck at 44%).
Per the bcv-commons/versification README, every scheme maps to the KJV standard, and the spine behaves as
KJV (empirically — protestant targets already align by identity). So for a spine ref R we fetch the target
verse via `from_standard`: the scheme's `standard_ref → source_ref` reverse map. protestant/unlisted →
identity, so existing languages are untouched.

Scheme per iso from `data/versification.json` (TEMP stopgap; later sourced from bcv-commons/bibles's
`versification` field). `orthodox`/`septuagint` → the `lxx` diff table (Orthodox = Septuagint tradition —
whether that's the right Synodal remap is validated empirically by the rus oracle after re-align).
"""
from __future__ import annotations

import json
from pathlib import Path

_VERSIF = Path("data/versification.json")
_REG_DIR = Path("resources/versification/schemes")
# scheme label → registry tsv basename. NOTE: `orthodox` is deliberately NOT mapped to `lxx`: the empirical
# test showed the Russian Synodal (labelled orthodox) is NOT Rahlfs-LXX-versified — its OT follows the
# Hebrew/Masoretic, so the LXX Psalm diffs made rus WORSE (44%→42%). orthodox needs its OWN registry scheme
# (a bcv-commons/versification ask); until then it falls back to identity. `septuagint` = genuine LXX.
_SCHEME_FILE = {"septuagint": "lxx", "lxx": "lxx", "hebrew": "hebrew"}  # → tsv basename
_IDENTITY = {"protestant", "kjv", ""}


def scheme_of(iso: str) -> str:
    if not _VERSIF.exists():
        return "protestant"
    cfg = {k: v for k, v in json.loads(_VERSIF.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    return cfg.get(iso, "protestant")


def _parse(ref: str):
    """'PSA 100:1' → ('PSA', 100, 1); None for title superscriptions / malformed."""
    try:
        book, cv = ref.split(" ")
        ch, v = cv.split(":")
        return (book, int(ch), int(v))
    except ValueError:
        return None


def load_reverse(scheme: str) -> dict:
    """{(book,ch,v)_KJV: (book,ch,v)_scheme} — from_standard. Empty (identity) for protestant/kjv/unknown."""
    if scheme in _IDENTITY:
        return {}
    fname = _SCHEME_FILE.get(scheme)
    if not fname:
        return {}
    fp = _REG_DIR / f"{fname}.tsv"
    if not fp.exists():
        return {}
    rev: dict[tuple, tuple] = {}
    with fp.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("source_ref"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src, std = _parse(parts[0]), _parse(parts[1])
            if src and std:
                rev[std] = src                                # from_standard[KJV ref] = scheme's ref
    return rev


def remapper(iso: str):
    """→ f(book, ch, v) mapping a spine (KJV) ref to the target's scheme ref; None if identity (protestant)."""
    rev = load_reverse(scheme_of(iso))
    if not rev:
        return None

    def f(book: str, ch: int, v: int):
        return rev.get((book, ch, v), (book, ch, v))
    return f
