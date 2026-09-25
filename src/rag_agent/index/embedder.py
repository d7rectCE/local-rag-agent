"""Text embedder on top of Hugging Face transformers.

For BGE-M3 one forward pass yields both the dense vector (CLS) and the sparse
lexical weights (``relu(sparse_linear(hidden))`` per token, max over repeated
tokens), as in the M3-Embedding paper (ТЗ [6], S5). Other encoders fall back to
dense-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_agent.config import EmbeddingConfig

log = logging.getLogger(__name__)


@dataclass
class Encoded:
    dense: np.ndarray  # (n, dim), float32, L2-normalised
    sparse: list[dict[int, float]] | None  # token id -> weight


def _find_sparse_head(model_name: str) -> Path | None:
    local = Path(model_name)
    if local.is_dir():
        path = local / "sparse_linear.pt"
        return path if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(model_name, "sparse_linear.pt"))
    except Exception:  # file absent for non-M3 models, or offline
        return None


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._tok = None
        self._model = None
        self._sparse = None
        self._special_ids: set[int] = set()
        self.device = "cpu"

    @property
    def name(self) -> str:
        return self.cfg.model

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.cfg.fp16 and device != "cpu" else torch.float32
        log.info("loading embedder %s on %s (%s)", self.cfg.model, device, dtype)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model)
        model = AutoModel.from_pretrained(self.cfg.model, dtype=dtype)
        self._model = model.to(device).eval()
        self.device = device
        self._special_ids = set(self._tok.all_special_ids)

        head = _find_sparse_head(self.cfg.model)
        if head is not None:
            linear = torch.nn.Linear(model.config.hidden_size, 1)
            linear.load_state_dict(torch.load(head, map_location="cpu", weights_only=True))
            self._sparse = linear.to(device=device, dtype=dtype).eval()

    @property
    def dim(self) -> int:
        self.load()
        return int(self._model.config.hidden_size)

    @property
    def has_sparse(self) -> bool:
        self.load()
        return self._sparse is not None

    def encode(self, texts: list[str], batch_size: int | None = None) -> Encoded:
        import torch

        self.load()
        n = len(texts)
        dense = np.zeros((n, self.dim), dtype=np.float32)
        sparse: list[dict[int, float]] | None = [{} for _ in range(n)] if self._sparse is not None else None
        order = sorted(range(n), key=lambda i: -len(texts[i]))  # similar lengths -> less padding
        bs = batch_size or self.cfg.batch_size
        i = 0
        while i < n:
            idx = order[i : i + bs]
            try:
                d, s = self._encode_batch([texts[j] for j in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("OOM while embedding, retrying with batch_size=%d", bs)
                continue
            dense[idx] = d
            if sparse is not None and s is not None:
                for k, j in enumerate(idx):
                    sparse[j] = s[k]
            i += len(idx)
        return Encoded(dense=dense, sparse=sparse)

    def _encode_batch(self, batch: list[str]) -> tuple[np.ndarray, list[dict[int, float]] | None]:
        import torch
        import torch.nn.functional as F

        enc = self._tok(batch, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if self.cfg.pooling == "cls":
                vec = hidden[:, 0]
            else:
                m = mask.unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
            dense = F.normalize(vec.float(), dim=-1).cpu().numpy()
            if self._sparse is None:
                return dense, None
            weights = (torch.relu(self._sparse(hidden)).squeeze(-1).float() * mask).cpu().numpy()
        ids = enc["input_ids"].cpu().numpy()
        sparse = []
        for row_ids, row_w in zip(ids.tolist(), weights.tolist()):
            vec_s: dict[int, float] = {}
            for tok, w in zip(row_ids, row_w):
                if w > 0 and tok not in self._special_ids and w > vec_s.get(tok, 0.0):
                    vec_s[tok] = w
            sparse.append(vec_s)
        return dense, sparse
