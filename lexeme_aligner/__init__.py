"""aligner — the Strong's word-alignment factory (offline producer).

Design: docs/aligner-plan.md. Generic pipeline (any language via a source adapter +
iso639-3); Indonesian is the pilot. Stage (a) = the $0 deterministic gloss-anchored
strategy; stage (b) = SimAlign neural fallback (added after the stage-(a) checkpoint).
Experiment artifacts go to aligner/out/ (gitignored); nothing writes to resources/
until it passes the benchmark gate.
"""
