"""Stage (c): neural alignment — SimAlign-style, multilingual contextual embeddings (LaBSE).

Aligns original-language tokens ↔ target tokens by cosine similarity of their sub-word embeddings,
then SimAlign's symmetrised extraction (mutual-argmax + optional itermax). No parallel-corpus training —
the multilingual encoder is the shared space; the aligner's job is just to read the similarity matrix.

Two layers, deliberately split so the logic is testable without the model:
  • `mutual_argmax` / `itermax` — PURE NUMPY matrix → alignment pairs. No torch. Unit-testable.
  • `NeuralAligner` — encodes tokens with LaBSE (transformers, the `[neural]` extra) and feeds the
    similarity matrix to the above. Heavy: torch + a ~1.8GB model download; CPU-slow but workable.

Independent of the gloss bootstrap (which is seeded from eflomal): the encoder is an outside signal, so
in the ensemble neural genuinely disagrees where eflomal/gloss are wrong — satisfying the handover's
independence rule.
"""
from __future__ import annotations

from lexeme_aligner.gloss_align import Match


def mutual_argmax(sim) -> list[tuple[int, int]]:
    """SimAlign 'inter': (i, j) aligned iff j is i's best target AND i is j's best source. High
    precision, lower recall. `sim` is a numpy [n_src, n_tgt] cosine matrix."""
    import numpy as np
    if sim.size == 0:
        return []
    src_best = sim.argmax(axis=1)                         # each source row's best target
    tgt_best = sim.argmax(axis=0)                         # each target col's best source
    return [(i, int(src_best[i])) for i in range(sim.shape[0]) if tgt_best[src_best[i]] == i]


def itermax(sim, max_iter: int = 2, threshold: float = 0.0) -> list[tuple[int, int]]:
    """SimAlign 'itermax': mutual-argmax, mask the matched rows/cols, repeat — recovers pairs a single
    mutual pass drops when two sources compete for one target. One-to-one; pairs score ≥ threshold.
    Matched rows/cols are tracked so a fully-masked row can't emit a spurious self-pair."""
    import numpy as np
    if sim.size == 0:
        return []
    work = sim.astype(float).copy()
    pairs: list[tuple[int, int]] = []
    used_i: set[int] = set()
    used_j: set[int] = set()
    for _ in range(max_iter):
        new = [(i, j) for i, j in mutual_argmax(work)
               if i not in used_i and j not in used_j and sim[i, j] >= threshold]
        if not new:
            break
        for i, j in new:
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
            work[i, :] = -np.inf                          # remove matched row + col from the next pass
            work[:, j] = -np.inf
    return pairs


class NeuralAligner:
    """LaBSE-encoded token aligner. Lazy-loads the model (optional `[neural]` extra)."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str = "auto",
                 threshold: float = 0.4, max_iter: int = 2, layer: int = 8, recall_fill: bool = False):
        # layer: SimAlign's key finding — an INTERMEDIATE encoder layer aligns far better than the last
        # (the last is tuned for the pretraining head / retrieval objective, not token correspondence).
        # recall_fill: mutual-argmax is high-precision but low-recall; after it, give each still-unaligned
        # source token its forward-argmax target (≥ threshold) — recovers recall the mutual pass drops.
        self.model_name = model_name
        self.device = self._resolve_device(device)         # "auto" → mps (Apple GPU) / cuda / cpu
        self.threshold, self.max_iter, self.layer, self.recall_fill = threshold, max_iter, layer, recall_fill
        self._tok = self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise SystemExit(f"[neural] needs torch + transformers — pip install -e '.[neural]' ({e})")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _embed(self, words: list[str]):
        """One embedding per word = mean of its sub-word piece vectors (last hidden state)."""
        import torch
        self._load()
        if not words:
            import numpy as np
            return np.zeros((0, 768), dtype="float32")
        enc = self._tok(words, is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
            layers = out.hidden_states                           # tuple: embeddings + one per layer
            L = min(self.layer, len(layers) - 1)                 # intermediate layer (SimAlign ~8)
            hidden = layers[L][0]                                 # [n_pieces, dim]
        word_ids = enc.word_ids(0)
        dim = hidden.shape[1]
        vecs = torch.zeros(len(words), dim)
        counts = torch.zeros(len(words), 1)
        for piece, wid in enumerate(word_ids):
            if wid is None:
                continue
            vecs[wid] += hidden[piece].cpu()
            counts[wid] += 1
        vecs = vecs / counts.clamp(min=1)
        vecs = torch.nn.functional.normalize(vecs, dim=1)
        return vecs.numpy()

    def align_verse(self, heb, tokens: list[str]) -> list[Match]:
        import numpy as np
        content = [h for h in heb if h.strong]
        if not content or not tokens:
            return []
        h_vec = self._embed([h.surface for h in content])
        t_vec = self._embed(tokens)
        sim = h_vec @ t_vec.T                              # cosine (both normalized)
        out: list[Match] = []
        used = set()
        for i, j in itermax(sim, self.max_iter, self.threshold):   # high-precision mutual core
            out.append(Match(content[i].idx, [j], round(float(sim[i, j]), 3), "neural"))
            used.add(i)
        if self.recall_fill:                               # forward-argmax for the rest (recall)
            for i in range(len(content)):
                if i in used:
                    continue
                j = int(sim[i].argmax())
                if sim[i, j] >= self.threshold:
                    out.append(Match(content[i].idx, [j], round(float(sim[i, j]), 3), "neural"))
        return sorted(out, key=lambda m: m.h_idx)

    def align_gap(self, heb, tokens: list[str], gap_idx: set, taken: set,
                  strong_surfaces: dict | None = None, anchors: dict | None = None,
                  lex_pos: dict | None = None, lex_translit: dict | None = None,
                  target_pos: dict | None = None, pos_weight: float = 0.2, strong_boost: float = 0.6,
                  name_boost: float = 0.6, pos_boost: float = 0.15) -> list[Match]:
        """Align ONLY the gap source tokens (`gap_idx`) onto the UNTAKEN targets, re-ranked by priors:

          • strong-rollup back-off (`strong_surfaces`, from the taken pool) — untaken target matching a
            known surface of the gap's Strong's: near-decisive (`strong_boost`, bypasses cosine floor).
          • name transliteration (`lex_translit` + `lex_pos`, prior-pack) — for pos=name gaps, an untaken
            target whose surface ≈ the romanized source (edit-distance): `name_boost`, bypasses floor.
          • grammatical (`target_pos` bootstrapped from taken pool × `lex_pos`) — soft `pos_boost` when
            the untaken target's inferred POS matches the gap's source POS.
          • positional/diagonal (`anchors`) — penalise distance from the interpolated expected position.

        strong/name matches are hi-conf; others still gated by the cosine floor. One target per gap token."""
        from lexeme_aligner.gloss_align import _name_score
        content = [h for h in heb if h.strong and h.idx in gap_idx]
        avail = [j for j in range(len(tokens)) if j not in taken]
        if not content or not avail:
            return []
        h_vec = self._embed([h.surface for h in content])
        t_vec = self._embed(tokens)
        sim = h_vec @ t_vec.T
        tnorm = [t.lower() for t in tokens]
        n_trg, n_src = len(tokens), max(len(heb), 1)
        order = {h.idx: k for k, h in enumerate(heb)}                # source token → ordinal position

        def expected(hidx: int) -> float:
            p = order.get(hidx, 0)
            if anchors:                                             # interpolate between nearest anchors
                below = [(order.get(a, 0), tp) for a, tp in anchors.items() if order.get(a, 0) <= p]
                above = [(order.get(a, 0), tp) for a, tp in anchors.items() if order.get(a, 0) >= p]
                b = max(below, default=None)
                a = min(above, default=None)
                if b and a and a[0] != b[0]:
                    return b[1] + (p - b[0]) / (a[0] - b[0]) * (a[1] - b[1])
                if b:
                    return b[1]
                if a:
                    return a[1]
            return p / n_src * n_trg                                # diagonal fallback

        scored = []
        for i, h in enumerate(content):
            exp = expected(h.idx)
            known = strong_surfaces.get(h.strong) if strong_surfaces else None
            spos = lex_pos.get(h.lexeme) if lex_pos else None
            translit = ((lex_translit.get(h.lexeme) or "").replace(".", "").replace("·", "")
                        if lex_translit else "")
            for j in avail:
                base = float(sim[i, j])
                is_strong = bool(known and tnorm[j] in known)
                is_name = bool(spos == "name" and translit and _name_score(translit, tokens[j]) >= 0.8)
                pos_ok = bool(spos and target_pos and target_pos.get(tnorm[j]) == spos)
                s = (base + (strong_boost if is_strong else 0.0) + (name_boost if is_name else 0.0)
                     + (pos_boost if pos_ok else 0.0) - pos_weight * abs(j - exp) / n_trg)
                scored.append((s, i, j, base, is_strong, is_name))
        scored.sort(key=lambda x: -x[0])
        out: list[tuple] = []                                       # (Match, prior) — prior tags the scorer
        done_src: set[int] = set()
        used: set[int] = set()
        for s, i, j, base, is_strong, is_name in scored:
            if i in done_src or j in used:
                continue
            hiconf = is_strong or is_name
            if not hiconf and base < self.threshold:               # strong/name bypass the cosine floor
                continue
            prior = "strong" if is_strong else "name" if is_name else "embedding"
            out.append((Match(content[i].idx, [j], 0.9 if hiconf else round(base, 3), "neural"), prior))
            done_src.add(i)
            used.add(j)
        return sorted(out, key=lambda mp: mp[0].h_idx)
