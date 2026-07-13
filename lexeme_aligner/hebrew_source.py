"""Hebrew (original) side — spine.db word tokens enriched with hbo.db per-occurrence data.

- shoresh/spine/spine.db `spine_words` is the alignment backbone: one row per UHB token,
  keyed (book, chapter, verse, idx), with surface / strong (bare int) / lemma / morph /
  is_content.
- resources/occurrences/hbo.db `occurrence` carries the per-occurrence BHSA layer:
  lex, stem (binyan), sp, English gloss, disambiguated sense + confidence.

The two tokenize differently (spine fuses prefixes: וַ⁠תֹּ֤אמֶר = conj+verb in ONE spine
token; BHSA splits them), so the join is STRONG-IN-ORDER within the verse — the nth spine
token bearing Strong's S matches the nth hbo row with Strong's S — not positional. This is
the pragmatic id-bridge from docs/aligner-plan.md §Design gotchas.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from lexeme_aligner.config import HBO_DB, SPINE_DB
from lexeme_aligner.refs import BOOK_NUMBERS, encode  # vendored — no cross-package import


def _derive_lexeme(padded_strong: str | None, lemma: str | None) -> str | None:
    """Stand-in lexeme until the lexeme-anchored spine lands: `<padded strong>|<lemma>`, which splits
    the homonyms a bare Strong's conflates (finer than strong, rolls up to it). Replaced verbatim by
    the spine's own `lexeme` column (MACULA lang+augmented-Strong's) when present."""
    if padded_strong and lemma:
        return f"{padded_strong}|{lemma}"
    return padded_strong or lemma


@dataclass
class HebToken:
    idx: int
    surface: str
    strong: str | None          # padded, e.g. H0559 (None: suffix pronouns etc.) — the ROLLUP
    lexeme: str | None          # the lexical ANCHOR: spine `lexeme` col if present, else strong|lemma
    lemma: str | None
    morph: str | None
    is_content: bool
    # from hbo.db (best-effort join):
    lex: str | None = None
    stem: str | None = None
    sp: str | None = None
    gloss_en: str | None = None
    sense: str | None = None
    sense_conf: float | None = None
    # filled by the aligner:
    matches: list = field(default_factory=list)


class HebrewSource:
    def __init__(self, spine_db: Path = SPINE_DB, hbo_db: Path = HBO_DB):
        self.spine = sqlite3.connect(f"file:{spine_db}?mode=ro", uri=True)
        # Forward-compat with the lexeme-anchored spine (docs/data-contracts.md): use the spine's own
        # `lexeme` column when it lands; until then derive a lexeme from (strong, lemma) so the rest of
        # the pipeline is already lexeme-anchored.
        _spine_cols = {r[1] for r in self.spine.execute("PRAGMA table_info(spine_words)")}
        self.has_lexeme = "lexeme" in _spine_cols
        # Enriched lexeme-spine carries gloss + disambiguated sense inline (bridge-joined upstream);
        # when present it's authoritative, so we read it directly and skip the hbo.db strong-in-order join.
        self.has_sense = "sense" in _spine_cols
        # hbo.db is the optional per-occurrence sense sidecar (sense-mining only).
        # Statistical methods (eflomal/IBM-1) need only spine + target USJ, so a
        # missing hbo.db must not be fatal — connect only when the file is present.
        self.hbo = (sqlite3.connect(f"file:{hbo_db}?mode=ro", uri=True)
                    if Path(hbo_db).exists() else None)

    def chapters(self, book: str) -> list[int]:
        return [r[0] for r in self.spine.execute(
            "SELECT DISTINCT chapter FROM spine_words WHERE book=? ORDER BY chapter", (book,))]

    def verses(self, book: str, chapter: int) -> list[int]:
        # verse 0 = psalm superscription (title); skip — not a content-alignment target,
        # and the encode()/versification handling treats titles separately (V1-gated).
        return [r[0] for r in self.spine.execute(
            "SELECT DISTINCT verse FROM spine_words WHERE book=? AND chapter=? AND verse>=1 "
            "ORDER BY verse", (book, chapter))]

    def verse_tokens(self, book: str, chapter: int, verse: int) -> list[HebToken]:
        toks: list[HebToken] = []
        pfx = "G" if BOOK_NUMBERS.get(book, 0) >= 40 else "H"   # NT=Greek(G), OT=Hebrew(H) strongs
        cur = self.spine.execute(
            "SELECT * FROM spine_words WHERE book=? AND chapter=? AND verse=? ORDER BY idx",
            (book, chapter, verse))
        names = [d[0] for d in cur.description]
        for raw in cur:
            r = dict(zip(names, raw))
            strong = r.get("strong")
            padded = f"{pfx}{int(strong):04d}" if strong else None
            lexeme = r.get("lexeme") if self.has_lexeme else _derive_lexeme(padded, r.get("lemma"))
            # Fused multi-token names (בֵּית לֶחֶם = 2 spine tokens, ONE Strong's, one BHSA
            # lexeme): merge consecutive same-strong tokens into one alignment unit — else
            # they inflate the denominator and double-consume target tokens. (Merge on the ROLLUP;
            # a fused name is one Strong's across differing lemmas.)
            if padded and toks and toks[-1].strong == padded:
                toks[-1].surface += " " + r.get("surface", "")
                continue
            tok = HebToken(r.get("idx"), r.get("surface"), padded, lexeme,
                           r.get("lemma"), r.get("morph"), bool(r.get("is_content")))
            if self.has_sense:                       # enriched spine: MACULA binyan + gloss + sense inline
                tok.stem = r.get("stem") or None     # MACULA binyan (qal/piel/hiphil). shoresh keys its
                tok.gloss_en = r.get("gloss") or None  # senses on (lexeme, stem, sense) — anchor is the
                tok.sense = r.get("sense") or None     # MACULA lexeme (BHSA `lex` dropped; CC-BY-clean key)
                tok.sense_conf = r.get("sense_conf")
            toks.append(tok)

        # strong-in-order join to hbo.db — only when the spine lacks inline sense AND the sidecar exists
        if self.has_sense or self.hbo is None:
            return toks
        ref = encode(book, chapter, verse)
        hbo_rows = list(self.hbo.execute(
            "SELECT lex,stem,sp,strong,gloss,sense,sense_conf FROM occurrence "
            "WHERE ref=? ORDER BY node", (ref,)))
        used = [False] * len(hbo_rows)
        for t in toks:
            if not t.strong:
                continue
            for i, (lex, stem, sp, strong, gloss, sense, conf) in enumerate(hbo_rows):
                if used[i] or strong != t.strong:
                    continue
                used[i] = True
                t.lex, t.stem, t.sp = lex, stem or None, sp
                t.gloss_en = gloss or None
                t.sense, t.sense_conf = sense or None, conf
                break
        return toks
