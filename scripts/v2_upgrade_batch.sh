#!/bin/bash
# v2 upgrade batch (NO publishing — user publishes afterwards).
#   Part 1  UPGRADE : re-align (eflomal → export lexeme-alignments → gloss) the 7 langs that lack t_idx
#                     (arb eng fra hau ind swe swk). The 5 already-current langs (asm ben hin rus spa)
#                     are kept as-is (fresh v2 builds with t_idx) — only neural-fed below.
#   Part 2  NEURAL  : gap_neural on ALL 12 (publish-safe prior-gate: strong+name, embedding dropped),
#                     bge-m3 on MPS. This is the long pole — left running.
#   Skips aligned_mwe (deferred); but the re-align gives t_idx so MWE is enabled for later.
cd "$(dirname "$0")/.." || exit 1
exec > out/_v2batch.log 2>&1
echo "=== V2 BATCH START $(date) ==="

reAlign() {  # iso dir scope name
  iso=$1; dir=$2; scope=$3; name=$4
  echo ">>> [upgrade $(date +%H:%M:%S)] $iso ($scope) dir=$dir"
  rm -f out/align_*_${iso}_*.jsonl                         # clean slate (old methods + ind stale numbering)
  python3 -m lexeme_aligner.run_pilot --method eflomal $scope --usj-dir "$dir" --iso "$iso" || echo "!! eflomal $iso FAILED"
  python3 -m lexeme_aligner.export_lex --iso "$iso" --method eflomal --lang-name "$name"     || echo "!! export  $iso FAILED"
  python3 -m lexeme_aligner.run_pilot --method gloss   $scope --usj-dir "$dir" --iso "$iso"  || echo "!! gloss   $iso FAILED"
  echo "<<< [upgrade] $iso done"
}

reAlign arb data/usj-arb       --ot  Arabic
reAlign eng data/usj-eng       --ot  English
reAlign fra data/usj-fra-lsg   --all French
reAlign hau data/usj-hau-ohcb  --all Hausa
reAlign ind data/usj-ind       --all Indonesian
reAlign swe data/usj-swe       --all Swedish
reAlign swk data/usj-swk       --all "Swedish Karnbibeln"

echo "=== PART 1 (upgrade) COMPLETE $(date) — starting neural ==="

neural() {  # iso dir scope
  iso=$1; dir=$2; scope=$3
  echo ">>> [neural $(date +%H:%M:%S)] $iso ($scope)"
  HF_HUB_OFFLINE=1 python3 -m lexeme_aligner.gap_neural --iso "$iso" $scope --usj-dir "$dir" \
    --neural-model BAAI/bge-m3 --neural-layer 16 --neural-device mps || echo "!! neural $iso FAILED"
  echo "<<< [neural] $iso done"
}

neural arb data/usj-arb       --ot
neural eng data/usj-eng       --ot
neural fra data/usj-fra-lsg   --all
neural hau data/usj-hau-ohcb  --all
neural ind data/usj-ind       --all
neural swe data/usj-swe       --all
neural swk data/usj-swk       --all
neural asm data/usj-asm       --all
neural ben data/usj-ben       --all
neural hin data/usj-hin       --all
neural rus data/usj-rus       --all
neural spa data/usj-spa       --all

echo "=== V2 BATCH COMPLETE $(date) ==="
