"""Cross-encoder reranker (ТЗ [8], S5): scores (query, passage) pairs jointly.
Default model: BAAI/bge-reranker-v2-m3 (multilingual, XLM-RoBERTa large)."""

from __future__ import annotations

import logging

import numpy as np

from rag_agent.config import RetrievalConfig
from rag_agent.index.embedder import enforce_hf_offline, load_pretrained

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, cfg: RetrievalConfig, device: str = "auto", fp16: bool = True, local_files_only: bool = True):
        self.model_name = cfg.rerank_model
        self.max_length = cfg.rerank_max_length
        self.batch_size = cfg.rerank_batch_size
        self._device_pref = device
        self._fp16 = fp16
        self._local_files_only = local_files_only
        self._tok = None
        self._model = None
        self.device = "cpu"

    @property
    def name(self) -> str:
        return self.model_name

    def load(self) -> None:
        if self._model is not None:
            return
        if self._local_files_only:
            enforce_hf_offline()  # before transformers is imported
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._fp16 and device != "cpu" else torch.float32
        log.info("loading reranker %s on %s", self.model_name, device)
        self._tok = load_pretrained(AutoTokenizer, self.model_name, self._local_files_only)
        model = load_pretrained(AutoModelForSequenceClassification, self.model_name, self._local_files_only, dtype=dtype)
        self._model = model.to(device).eval()
        self.device = device

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance scores in [0, 1] (sigmoid of the logit), one per passage."""
        import torch

        self.load()
        scores = np.zeros(len(passages), dtype=np.float32)
        order = sorted(range(len(passages)), key=lambda i: -len(passages[i]))
        bs = self.batch_size
        i = 0
        while i < len(order):
            idx = order[i : i + bs]
            enc = self._tok(
                [query] * len(idx), [passages[j] for j in idx],
                padding=True, truncation="only_second", max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            try:
                with torch.inference_mode():
                    logits = self._model(**enc).logits.view(-1).float()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                continue
            scores[idx] = torch.sigmoid(logits).cpu().numpy()
            i += len(idx)
        return scores
