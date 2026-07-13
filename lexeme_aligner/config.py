"""Config — every external path in ONE place, env-overridable.

This is the aligner's only coupling to its host layout. Defaults point at the in-repo `data/` dir
(spine.db is copied there, gitignored — see data/PROVENANCE.txt); set the env vars to run against a
different spine, published bcv-commons datasets, or a spine built from STEPBible+MACULA. See DATA.md.

  ALIGNER_SPINE_DB   original-language backbone (spine_words: book,chapter,verse,idx,surface,strong,lemma,morph,is_content)
  ALIGNER_HBO_DB     per-occurrence sense sidecar (occurrence: ref,lex,stem,sp,strong,gloss,sense,sense_conf) — optional
  ALIGNER_RESOURCES  dir holding gloss priors (word_glosses/, llm_strongs_glosses/, strongs_tw.tsv, tw_articles/) — optional
  ALIGNER_OUT        experiment output dir (gitignored)
"""
from __future__ import annotations

import os
from pathlib import Path

# repo root = two levels up from this file (lexeme_aligner/ under the repo). Defaults point at the
# in-repo data/ dir (spine.db copied there, gitignored); override any of these via the env vars.
_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v) if v else default


SPINE_DB = _p("ALIGNER_SPINE_DB", _DATA / "lexeme-spine.db")  # required — lexeme-anchored (see data/PROVENANCE.txt)
HBO_DB = _p("ALIGNER_HBO_DB", _DATA / "hbo.db")            # optional — per-occurrence sense sidecar
RESOURCES = _p("ALIGNER_RESOURCES", _DATA / "resources")  # optional — gloss priors (bcv-commons/strongs)
OUT = _p("ALIGNER_OUT", _ROOT / "out")                    # experiment output (gitignored)
LEX_ROOT = _p("ALIGNER_LEX_ROOT", _ROOT / "lexeme-alignments")  # published dataset root (was aligned_lex)
# language-independent prior pack pulled from bcv-commons/prior-pack (HF, CC-BY) — feeds the recipes
# (R1 keyness-filter, R2 sense-surface, R3 gap-map, LXX NT-gap). See internal-docs/aligner-handover.md.
PRIOR_PACK = _p("ALIGNER_PRIOR_PACK", _ROOT / "resources" / "prior-pack" / "prior_pack.parquet")
