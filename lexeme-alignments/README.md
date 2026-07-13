---
pretty_name: Lexeme-anchored alignments (surface → lexeme, Strong's-bridged)
language:
  - arb
  - asm
  - ben
  - eng
  - fra
  - hau
  - hin
  - ind
  - rus
  - spa
  - swe
  - swk
tags:
  - bible
  - word-alignment
  - lexeme
  - strongs
  - interlinear
task_categories:
  - translation
  - token-classification
license: cc0-1.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: iso=*/data.parquet
---

# lexeme-alignments — surface → original-language lexeme (Strong's-bridged)

For each language, the attested mapping from target **surface word-forms** → the original-language
**lexeme** they render, mined by the aligner. **Lexeme-anchored, provenance-honest, additive** — the
design principles are in [`docs/publishing-principles.md`](../docs/publishing-principles.md). One
language per partition, for consumption by **bcv-commons** and downstream tools.

> The `language:` list above tracks the published partitions; the authoritative list is always
> `manifest.json`.

## The anchor: lexeme, not Strong's
The anchor of record is the **MACULA lexeme** (`hbo:0430`, `grc:2316`) — the precise dictionary unit.
The bare **Strong's** number is coarser (it conflates homonyms and sense-splits — one Strong's rolls up
several lexemes), so it rides along as a **bridge column** (`strong`), never the key. Lexeme-precise
consumers use `lexeme`; Strong's-ecosystem tools use the derived on-ramp (below). Nothing is hidden —
the coarse key is a convenience, the fine key is the truth.

## Schema (per row)
| column | type | meaning |
|---|---|---|
| `surface` | string | target rendering, lowercased (content tokens; may be multi-word) |
| `lexeme` | string | **the anchor** — MACULA lexical id (`lang:augmented-strong`) |
| `strong` | string | Strong's number (`H####`/`G####`) — the **rollup** bridge of `lexeme` |
| `method` | string | **which method attested this pair** — `eflomal` / `gloss` / `neural` |
| `base_text` | string | **which edition** the surface is from (e.g. `BSB`, `eng_ylt`) |
| `count` | int32 | times this (surface → lexeme) was aligned **in that method + edition** |
| `share` | float32 | `count / Σ count for that surface`, **within (method, base_text)** = P(lexeme \| surface) |
| `hi_conf` | float32 | fraction of this pair's alignments that were intersection-backed (score ≥ 0.9) |

`iso` is recovered from the Hive partition path (`iso=<iso>/`). Two honest provenance axes: `method`
(*how* aligned) and `base_text` (*which* edition).

### It's an additive union — nothing is merged away
Rows are the **union of the methods**, each tagged with its `method`. A surface→lexeme attested by both
eflomal and neural is **two rows** (eflomal ×N, neural ×M) — full provenance, no winner-take-all merge.
This means:
- a **neural-only** fact says `method=neural` — it can never masquerade as eflomal/gloss-attested;
- an **enhanced translation** that renders one lexeme with many words keeps all of them — we never force
  a lexeme to a single "canonical" surface;
- `count`s are **per-method**, so *do not sum across methods* to get an occurrence total (the same verse
  is often aligned by more than one method — that would double-count; see the on-ramp script).

### Multiple editions of one language (pooling)
Some languages ship several editions pooled into one partition, each row tagged by `base_text`
(e.g. `eng` = BSB + YLT; `arb` = Van Dyck + New Arabic Version; `swe` = Folkbibeln + Kärnbibeln). This
is additive — every edition's renderings are kept, and:
- **single edition** → filter `base_text = '<edition>'`;
- **cross-edition agreement** (a strong confidence signal) → a surface→lexeme attested by **more than one**
  `base_text` is corroborated across independent translations; derive it by counting distinct `base_text`
  per (surface, lexeme). An enhanced/literal edition (e.g. YLT's `begat`/`begotten`) contributes its own
  renderings without overwriting the others.
- **takedown** → if a rights-holder objects, drop that `base_text`'s rows and republish (content-addressed);
  never re-emit provenance-stripped.

The `manifest.json` entry lists the pooled `base_texts`, per-edition row counts (`by_base_text`), and a
`sources` pointer per edition.

## Using the data — pick your operating point
Three independent signals; combine them. The dataset ships the full distribution rather than
pre-filtering, so precision / coverage / provenance are sliders **you** control:

| goal | filter |
|---|---|
| everything / max recall | all rows |
| **exclude the neural gap-filler** | `method != 'neural'` |
| one edition only | `base_text == '<edition>'` |
| cross-edition-corroborated | keep (surface, lexeme) with ≥2 distinct `base_text` |
| balanced (recommended default) | argmax-`share` per (surface, method), `count ≥ 2` |
| high precision | `hi_conf ≥ 0.5`, `count ≥ 2` |
| one method only | `method == 'eflomal'` (or `gloss`) |

- **`share`** → *which lexeme* (P(lexeme\|surface) within a method).
- **`hi_conf`** → *how reliable the placement* (intersection-backed share).
- **`count`** → *how much evidence* (`count: 1` rests on a single occurrence).

## Derived views (example scripts, never a second source of truth)
1. **Strong's on-ramp** — roll `lexeme→strong` from a single base method into a clean Strong's-keyed
   table (surface + frequency), for ecosystem tools:
   ```bash
   python3 scripts/strongs_view.py --iso swe                    # → out/strongs_view_swe.tsv
   python3 scripts/strongs_view.py --iso swe --hi-conf 0.5 --min-share 0.02
   ```
   It picks one base method (default `eflomal`) so per-method counts don't double-count, then aggregates
   per (strong, surface) with `share = P(surface | strong)`.
2. **Merged best-pick** (optional, lossy) — a single-answer-per-token convenience, regenerable from the
   union via the contest-rule merge:
   ```bash
   python3 -m lexeme_aligner.merge_align --iso swe --methods eflomal,gloss,neural \
     --contest-rule data/contest_rule.json      # → align_merged_swe_*.jsonl, then export --methods merged
   ```
   Labelled **lossy** on purpose — it drops valid alternatives; use it only when you want exactly one row.

## Layout — why the bulk data isn't in git
```
lexeme-alignments/
  README.md            # committed — this file
  manifest.json        # committed — per-language metadata + content hash (the durable record)
  iso=<iso>/           # GIT-IGNORED — bulk data, published out-of-band (HF / object storage)
    data.parquet
```
`manifest.json` is git's small, diffable record of what exists and what it hashes to; each partition is
keyed by its `content_sha256`.

## Provenance & quality
Per-language provenance (methods present, per-method row counts, testament, counts, `hi_conf_ge_0.9`,
spine tags, content hash) lives in `manifest.json`. Every language is produced by the **same pipeline**,
validated against Clear-Bible manual gold **where it exists** — token-weighted top-1 of ~92–97%
(Strong's grain) / ~89–92% (lexeme grain — the anchor's headline; `docs/benchmark.md`). Languages without
usable gold (`ind`; `rus`, whose only manual reference is itself mis-aligned) run the **identical**
pipeline and are **not lower quality — simply un-cross-checked**. We do **not** stamp a verified/unverified
tier. Your confidence signal is the same for every language: the row-level `method` / `hi_conf` / `share` /
`count`. These are raw aligned counts, not hand-checked.

## Reproducibility (content-addressed)
The statistical aligner (eflomal) seeds from `/dev/urandom`, so regeneration varies ~1% run-to-run. This
is a **content-addressed release**: inputs are pinned (spine tags + each source text's `sha256`,
`data/pins/`), and each partition is fixed by its **`content_sha256`** in `manifest.json` — that hash
*is* the identity of what was released. Consume a specific release by its hash; a rebuild won't match
byte-for-byte.

## Authentication & publishing (one-time)
```bash
python3 -c "from huggingface_hub import login; login()"        # cached → ~/.cache/huggingface/token
python3 -m lexeme_aligner.export_lex --iso <iso> --lang-name <Name> \
  --publish bcv-commons/lexeme-alignments --create
```
Use a fine-grained write token scoped to the target dataset. The push uploads only this language's
partition + the shared `manifest.json`/`README.md`; other languages are untouched.

## License
**This catalogue is CC0-1.0.** It is *derived, factual* data — lexeme ids, Strong's rollups, alignment
counts, `share`/`hi_conf` statistics, method tags, and a de-arranged type-level list of word forms. It
does **not** reproduce the running text of any translation (no verse refs, no word order), so the
copyrightable expression of the sources is not present.

Each `surface` is nonetheless a word form from a **source translation**, and those keep their **own**
licenses. Every language's `manifest.json` entry carries a `source` **pointer**
(`provider`/`edition`/`license_url`) — follow it for the authoritative terms. Pointing to a source does
not by itself grant permission to derive from it; for any source whose terms restrict derivatives, obtain
that separately.
