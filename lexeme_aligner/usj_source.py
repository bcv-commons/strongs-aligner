"""USJ source adapter — read a book's USJ, emit per-verse target-language tokens.

The only format-specific code in the pipeline (docs/aligner-plan.md §Generic input):
walk the USJ `content` tree in document order, track chapter/verse markers, collect
translatable text, and EXCLUDE apparatus by element type/marker (notes, headings,
titles, intro material) so footnote/heading words can never leak into the alignment.

Tokenization is generic unicode word-splitting; language-specific normalization
(e.g. Indonesian clitic stripping) lives in gloss_align, not here.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

# Non-scripture paragraph markers (digits stripped before lookup): headers, titles,
# section headings, psalm titles (d), speaker lines, intro material. `b` is a blank line.
_SKIP_PARA = {
    "h", "toc", "mt", "imt", "ms", "mr", "s", "sr", "r", "d", "sp", "sd",
    "cl", "cp", "ca", "ide", "rem", "b", "ib", "ip", "is", "io", "iot",
    "ili", "im", "imi", "ipi", "iq", "ie", "periph", "restore", "lit",
}
# Character markers whose text is NOT translation text (alternate chapter/verse numbers).
_SKIP_CHAR = {"va", "vp", "ca", "cp", "fv", "fm"}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def strip_marks(text: str) -> str:
    """NFC, then drop combining marks (Unicode Mn) — Arabic harakat, Hebrew points, etc. Otherwise the
    word regex shatters a diacritized word at every mark (فِي → ف, ي). Precomposed Latin accents (é, å)
    survive NFC as single code points, so French/Swedish are unaffected; only decomposed marks go."""
    return "".join(c for c in unicodedata.normalize("NFC", text) if not unicodedata.combining(c))


def _base_marker(marker: str) -> str:
    return (marker or "").rstrip("0123456789")


def read_verses(usj_path: Path, warn=sys.stderr) -> dict[tuple[int, int], str]:
    """{(chapter, verse): text} for one book's USJ file. Verse ranges ("1-2") are
    keyed by their first number. Unknown para markers are included but logged once."""
    usj = json.loads(Path(usj_path).read_text(encoding="utf-8"))
    verses: dict[tuple[int, int], list[str]] = {}
    state = {"ch": 0, "v": 0}
    seen_unknown: set[str] = set()

    def emit(text: str) -> None:
        if state["ch"] and state["v"] and text:
            verses.setdefault((state["ch"], state["v"]), []).append(text)

    def walk(items) -> None:
        for it in items:
            if isinstance(it, str):
                emit(it)
                continue
            t = it.get("type")
            if t == "chapter":
                state["ch"] = int(re.match(r"\d+", str(it.get("number", "0"))).group())
                state["v"] = 0
            elif t == "verse":
                m = re.match(r"\d+", str(it.get("number", "0")))
                state["v"] = int(m.group()) if m else 0
            elif t == "note":
                continue                                    # footnotes / cross-refs: never text
            elif t == "para":
                base = _base_marker(it.get("marker", ""))
                if base in _SKIP_PARA:
                    continue
                if base not in {"p", "m", "po", "pr", "cls", "pmo", "pm", "pmc", "pmr",
                                "pi", "mi", "nb", "pc", "ph", "q", "qr", "qc", "qa",
                                "qac", "qm", "qd", "lh", "li", "lf", "lim", "tr", "tc",
                                "tcr", "th", "thr"} and base not in seen_unknown:
                    seen_unknown.add(base)
                    print(f"[usj] note: unknown para marker '{it.get('marker')}' — included",
                          file=warn)
                if it.get("content"):
                    walk(it["content"])
            elif t == "char":
                if _base_marker(it.get("marker", "")) in _SKIP_CHAR:
                    continue
                if it.get("content"):
                    walk(it["content"])
            elif it.get("content"):
                walk(it["content"])

    walk(usj.get("content", []))
    return {k: " ".join(parts) for k, parts in verses.items()}


def tokenize(text: str) -> list[str]:
    """Generic unicode word tokens, order preserved (normalization happens later). A token is a run of
    letters + their combining marks (a grapheme cluster), so Indic/diacritized scripts tokenize by WORD,
    not shatter at every vowel sign — दाऊद stays one token, not द+ऊद (the old `[^\\W\\d_]+` regex broke at
    spacing marks like the Mc matra ा). `strip_marks` still drops nonspacing marks (Arabic harakat, Hebrew
    points, Indic viramas) for match-normalisation; spacing marks stay inside the cluster. Latin/Cyrillic
    unaffected (no combining marks after NFC)."""
    toks: list[str] = []
    cur: list[str] = []
    for ch in strip_marks(text):
        if unicodedata.category(ch)[0] in ("L", "M"):        # letter or combining mark → part of the word
            cur.append(ch)
        elif cur:
            toks.append("".join(cur))
            cur = []
    if cur:
        toks.append("".join(cur))
    return toks
