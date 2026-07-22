"""Cross-source language/edition discovery — cdn.bibel.wiki's catalog-index.json +
catalog-overlap.json, covering PKF (`p`), helloAO (`h`), and DBT (`d`) in one place.

WHAT THIS IS: catalog-index.json is a flat [iso, testament, source, edition_count] list across all
three sources (1,876 distinct languages). catalog-overlap.json adds a `defaults` pick per
iso:testament (following priority [pkf, helloao, dbt]) plus, where multiple sources carry the SAME
language, an actual TEXT-SIMILARITY comparison ("identical", 1.0 — derived from probe-verse
fingerprinting, e.g. REV15/PSA117/PSA51, the same spirit as our own versification.py structural
fingerprinting) — telling us whether a second source's edition is the same underlying text (skip, to
avoid double-counting) or a genuinely different one (safe to pool, matching the existing
eng=BSB+eng_ylt / arb=arb_vdv+ARBNAV pattern).

WHAT THIS UNLOCKS: for the 1,109 catalog languages absent from our own gold_langs/aligned set but
reachable via pkf or helloao (sources we already have adapters for — cdn_source.py / helloao_source.py),
`resolve()` gives a ready-to-use ingest plan (source + the EXACT parameter each adapter needs — pkf
just needs --iso; helloAO needs the precise translation id, e.g. "aai_wbt", which only this catalog's
comparison data reliably supplies) — no per-language manual research needed.

WHAT THIS DOES NOT UNLOCK: DBT-only languages (~755 more). Investigated directly (session-time,
2026-07-21) — cdn.bibel.wiki exposes DBT DISCOVERY metadata only (this catalog + dbt/_catalog.json +
dbt/<iso>/media.json's fileset routing), not actual fetchable verse text; a dozen plausible endpoint
patterns (fileset-id-based, book-based, timing/caption-based) all 404, no browsable app either. That
content most likely lives behind Faith Comes By Hearing's own DBP/DBT API, which needs registration +
an API key — a credentials decision for the user, not something to route around silently. `resolve()`
surfaces this case explicitly (source="dbt", fetchable=False) rather than pretending to support it.

No git-commit anchor exists for this data (server-generated, no `generated_at`) — pinned by content
sha256 instead (same discipline cdn_source.py already uses for its own PKF payload verification).

    python3 -m lexeme_aligner.catalog_source --fetch                    # pin the catalog locally
    python3 -m lexeme_aligner.catalog_source --resolve swa --testament nt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://cdn.bibel.wiki/dbt/_app"
_UA = "lexeme-aligner/0.1 (+https://github.com/bcv-commons/lexeme-aligner)"
_DIR = Path("data/dbt_catalog")
_FILES = {"index": "catalog-index.json", "overlap": "catalog-overlap.json"}
PRIORITY = ("pkf", "helloao", "dbt")
_SOURCE_LETTER = {"p": "pkf", "h": "helloao", "d": "dbt"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310 — fixed https CDN origin
        return r.read()


def fetch(dir_: Path = _DIR) -> dict:
    """Download + content-hash-pin both catalog files. Idempotent-ish (always re-fetches — this is a
    live service index, not a versioned release; the pin records what we got, for provenance/drift
    detection, not to skip a re-download the way commit-pinned fetches do)."""
    dir_.mkdir(parents=True, exist_ok=True)
    pin = {"provider": "cdn.bibel.wiki/dbt", "files": {}}
    for key, fname in _FILES.items():
        data = _get(f"{BASE}/{fname}")
        (dir_ / fname).write_bytes(data)
        pin["files"][fname] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        print(f"[catalog_source] {fname}: {len(data)} bytes, sha256={pin['files'][fname]['sha256'][:12]}…",
              file=sys.stderr)
    (dir_ / "pin.json").write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pin


def load(dir_: Path = _DIR) -> tuple[dict, dict]:
    idx_fp, ov_fp = dir_ / _FILES["index"], dir_ / _FILES["overlap"]
    if not idx_fp.exists() or not ov_fp.exists():
        fetch(dir_)
    return (json.loads(idx_fp.read_text(encoding="utf-8")),
            json.loads(ov_fp.read_text(encoding="utf-8")))


def resolve(iso: str, testament: str = "nt", dir_: Path = _DIR) -> dict | None:
    """iso + testament -> an ingest plan: {source, fetchable, param, edition_code, alternates}.
    `param` is the exact value the corresponding adapter needs (pkf: the iso itself; helloao: the
    translation id, e.g. "aai_wbt"; dbt: None, not fetchable here). `alternates` lists OTHER sources
    that carry a genuinely DIFFERENT edition (not "identical" per the overlap comparison) — candidates
    for edition-pooling, same pattern as eng=BSB+eng_ylt."""
    _, overlap = load(dir_)
    matches = [e for e in overlap["entries"] if e[0] == iso and e[1] == testament]
    if not matches:
        return None

    default_key = f"{iso}:{testament}"
    default_source = overlap["defaults"].get(default_key)
    chosen = next((e for e in matches if e[2] == default_source), None) or matches[0]

    source, edition_code = chosen[2], chosen[3]
    comparisons = chosen[4] if len(chosen) > 4 else []
    source_id = chosen[5] if len(chosen) > 5 else None

    alternates = []
    for other_src, verdict, score in comparisons:
        other_source = other_src.split(":", 1)[0]
        if verdict != "identical":
            alternates.append({"source": other_source, "ref": other_src, "verdict": verdict, "score": score})

    if source == "pkf":
        param = iso
    elif source == "helloao":
        param = source_id
    else:
        param = None

    return {
        "iso": iso, "testament": testament, "source": source, "fetchable": source != "dbt",
        "param": param, "edition_code": edition_code,
        "alternates": alternates,
        "note": (None if source != "dbt" else
                 "DBT-native text is not fetchable via this CDN — needs Faith Comes By Hearing's "
                 "DBP/DBT API (registration + API key required, not currently configured)."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download + pin the catalog")
    ap.add_argument("--resolve", metavar="ISO", default=None)
    ap.add_argument("--testament", choices=["nt", "ot"], default="nt")
    ap.add_argument("--dir", type=Path, default=_DIR)
    args = ap.parse_args()

    if args.fetch:
        fetch(args.dir)
    if args.resolve:
        plan = resolve(args.resolve, args.testament, args.dir)
        if plan is None:
            print(f"[catalog_source] no {args.testament} entry for '{args.resolve}' in the catalog",
                  file=sys.stderr)
            return 1
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.fetch and not args.resolve:
        ap.error("need --fetch and/or --resolve")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
