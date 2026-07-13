"""Pre-filter the source lexemes/occurrences left WITHOUT a usable signal by eflomal + gloss.

Benchmarks showed the full ensemble never beats gloss: neural votes everywhere and its correlated,
lower-precision calls out-vote gloss's better ones. The fix is to run neural ONLY on the gaps — the
source tokens neither eflomal nor gloss aligned — so it can add coverage but never override a mode that
already has signal.

A source content token `(ref, h_idx)` is COVERED if eflomal OR gloss aligned it to a non-empty target
with score ≥ `min_score`. The GAP = spine content tokens (in the aligned books) that neither covered:
  • occurrence gap — specific (ref, h_idx) to target
  • lexeme gap     — lexemes NEVER covered anywhere (zero signal at all)

    python3 -m lexeme_aligner.neural_gap --iso fra                    # report the gap
    python3 -m lexeme_aligner.neural_gap --iso fra --emit out/gap_fra.json  # + write target refs/idx
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from lexeme_aligner.config import OUT, SPINE_DB
from lexeme_aligner.hebrew_source import HebrewSource
from lexeme_aligner.refs import encode


def _books_present(iso: str, out_dir: Path, method: str) -> set[str]:
    pre = f"align_{method}_{iso}_"
    return {fp.stem[len(pre):] for fp in out_dir.glob(f"{pre}*.jsonl")}


def compute_gaps(iso: str, out_dir: Path, spine: HebrewSource,
                 methods: tuple[str, ...], min_score: float):
    """→ (all_lex_occ, covered_lex, gap_occ_per_lex, gap_refs, n_covered_occ). Counts are occurrences."""
    covered: set[tuple[int, int]] = set()          # (ref, h_idx) aligned by SOME mode
    covered_lex: set[str] = set()
    for m in methods:
        for fp in sorted(out_dir.glob(f"align_{m}_{iso}_*.jsonl")):
            with fp.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    for p in rec["pairs"]:
                        if (p.get("content") and (p.get("target") or "").strip()
                                and (p.get("score") or 0) >= min_score):
                            covered.add((rec["ref"], p["h_idx"]))
                            if p.get("lexeme"):
                                covered_lex.add(p["lexeme"])

    books = _books_present(iso, out_dir, methods[0])   # coverage universe = books the base method ran on
    if not books:
        raise SystemExit(f"no align_{methods[0]}_{iso}_*.jsonl under {out_dir}")
    all_lex_occ: collections.Counter = collections.Counter()   # lexeme -> total content occurrences
    gap_occ_per_lex: collections.Counter = collections.Counter()  # lexeme -> uncovered occurrences
    gap_refs: dict[int, list[int]] = collections.defaultdict(list)  # ref -> [h_idx,…] to give neural
    for book in sorted(books):
        for ch in spine.chapters(book):
            for v in spine.verses(book, ch):
                ref = encode(book, ch, v)
                for t in spine.verse_tokens(book, ch, v):
                    if not (t.is_content and t.lexeme):
                        continue
                    all_lex_occ[t.lexeme] += 1
                    if (ref, t.idx) not in covered:
                        gap_occ_per_lex[t.lexeme] += 1
                        gap_refs[ref].append(t.idx)
    return all_lex_occ, covered_lex, gap_occ_per_lex, gap_refs, len(covered)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--methods", default="eflomal,gloss", help="modes that count as 'covered'")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="a pair counts as signal only at/above this score (raise to treat low-conf as gap)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--spine-db", type=Path, default=SPINE_DB)
    ap.add_argument("--emit", type=Path, default=None, help="write {ref: [h_idx,…]} gap targets as JSON")
    args = ap.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(","))
    spine = HebrewSource(spine_db=args.spine_db)
    all_lex_occ, covered_lex, gap_per_lex, gap_refs, n_cov = compute_gaps(
        args.iso, args.out, spine, methods, args.min_score)

    tot_occ = sum(all_lex_occ.values())
    gap_occ = sum(gap_per_lex.values())
    tot_lex = len(all_lex_occ)
    lex_never = [lx for lx in all_lex_occ if lx not in covered_lex]      # zero signal anywhere
    print(f"[gap] {args.iso} · covered-by {methods} (min_score {args.min_score})", file=sys.stderr)
    print(f"  content occurrences: {tot_occ}  ·  covered {n_cov}  ·  GAP {gap_occ} "
          f"({100*gap_occ/max(1,tot_occ):.1f}%)", file=sys.stderr)
    print(f"  content lexemes:     {tot_lex}  ·  {tot_lex-len(lex_never)} with signal  ·  "
          f"{len(lex_never)} NEVER aligned ({100*len(lex_never)/max(1,tot_lex):.1f}%)", file=sys.stderr)
    print(f"  gap verses (≥1 gap token): {len(gap_refs)}  → the neural target scope", file=sys.stderr)
    print("  top never-aligned lexemes by frequency:", file=sys.stderr)
    for lx, n in sorted(((lx, all_lex_occ[lx]) for lx in lex_never), key=lambda x: -x[1])[:8]:
        print(f"    {lx:12} ×{n}", file=sys.stderr)

    if args.emit:
        args.emit.write_text(json.dumps({str(r): idxs for r, idxs in gap_refs.items()}), encoding="utf-8")
        print(f"  → wrote {len(gap_refs)} gap verses to {args.emit}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
