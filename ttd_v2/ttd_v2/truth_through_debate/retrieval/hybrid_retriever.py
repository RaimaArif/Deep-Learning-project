"""
retrieval/hybrid_retriever.py

Four retrieval backends running in parallel (asyncio.gather).
Results are merged, deduplicated, and re-ranked by score.

Backends
--------
WikipediaRetriever   — Wikipedia REST API, no key needed
TavilyRetriever      — Real-time web search (requires TAVILY_API_KEY)
BM25Retriever        — TF-IDF / BM25 over FEVER corpus sentences
FAISSRetriever       — Dense vector search via sentence-transformers + FAISS
"""
from __future__ import annotations
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any

import aiohttp
import requests

from truth_through_debate.config import Config
from truth_through_debate.schema import Evidence

log = logging.getLogger(__name__)


# ── Wikipedia ─────────────────────────────────────────────────────────────────

class WikipediaRetriever:
    BASE = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "TruthThroughDebate/2.0 (research; python-aiohttp)"}

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def retrieve(self, claim: str, session: aiohttp.ClientSession) -> list[Evidence]:
        try:
            params = {"action":"query","list":"search","srsearch":claim,
                      "srlimit":self.cfg.top_k_per_source,"format":"json","utf8":1}
            async with session.get(self.BASE, params=params, headers=self.HEADERS,
                                   timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json(content_type=None)
            results = data.get("query", {}).get("search", [])

            evidence = []
            for res in results:
                title = res["title"]
                extract_params = {"action":"query","prop":"extracts","exsentences":4,
                                  "exintro":True,"explaintext":True,"titles":title,"format":"json"}
                async with session.get(self.BASE, params=extract_params, headers=self.HEADERS,
                                       timeout=aiohttp.ClientTimeout(total=8)) as r2:
                    d2 = await r2.json(content_type=None)
                pages = d2.get("query", {}).get("pages", {})
                for page in pages.values():
                    text = page.get("extract", "").strip()
                    if text:
                        evidence.append(Evidence(
                            text=text[:600], source="wikipedia",
                            title=title, url=f"https://en.wikipedia.org/wiki/{title.replace(' ','_')}",
                            score=1.0 - (len(evidence) * 0.05),
                        ))
                await asyncio.sleep(0.05)
            return evidence
        except Exception as e:
            log.warning("[Wikipedia] %s", e)
            return []


# ── Tavily ────────────────────────────────────────────────────────────────────

class TavilyRetriever:
    URL = "https://api.tavily.com/search"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def retrieve(self, claim: str, session: aiohttp.ClientSession) -> list[Evidence]:
        if not self.cfg.use_tavily or not self.cfg.tavily_api_key:
            return []
        try:
            payload = {"api_key": self.cfg.tavily_api_key, "query": claim,
                       "search_depth": "basic", "max_results": self.cfg.top_k_per_source,
                       "include_answer": False}
            async with session.post(self.URL, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=12)) as r:
                r.raise_for_status()
                data = await r.json()
            ev = []
            for res in data.get("results", []):
                content = res.get("content", "").strip()
                if content:
                    ev.append(Evidence(
                        text=content[:600], source="tavily",
                        title=res.get("title", ""), url=res.get("url", ""),
                        score=res.get("score", 0.5),
                    ))
            return ev
        except Exception as e:
            log.warning("[Tavily] %s", e)
            return []


# ── BM25 (FEVER corpus) ───────────────────────────────────────────────────────

class BM25Retriever:
    """
    Loads FEVER wiki-pages corpus and runs BM25 ranking.
    Index is built lazily on first call and cached in memory.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._bm25: Any = None
        self._corpus: list[dict] = []

    def _lazy_load(self) -> None:
        if self._bm25 is not None:
            return
        if not self.cfg.use_bm25:
            return
        path = Path(self.cfg.fever_jsonl_path)
        if not path.exists():
            log.warning("[BM25] FEVER file not found at %s — BM25 disabled.", path)
            self.cfg.use_bm25 = False
            return
        try:
            from rank_bm25 import BM25Okapi
            import json as _json
            corpus_texts, docs = [], []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    obj = _json.loads(line)
                    claim = obj.get("claim", "")
                    if claim:
                        corpus_texts.append(claim.lower().split())
                        docs.append({"text": claim, "title": "", "source": "bm25_fever"})
            self._corpus = docs
            self._bm25 = BM25Okapi(corpus_texts)
            log.info("[BM25] Indexed %d FEVER sentences.", len(docs))
        except Exception as e:
            log.warning("[BM25] Load error: %s", e)
            self.cfg.use_bm25 = False

    async def retrieve(self, claim: str, _session: Any) -> list[Evidence]:
        if not self.cfg.use_bm25:
            return []
        self._lazy_load()
        if self._bm25 is None:
            return []
        try:
            tokens = claim.lower().split()
            scores = self._bm25.get_scores(tokens)
            import numpy as np
            top_idx = np.argsort(scores)[::-1][:self.cfg.top_k_per_source]
            return [
                Evidence(text=self._corpus[i]["text"][:600], source="bm25_fever",
                         title="FEVER corpus", score=float(scores[i]))
                for i in top_idx if scores[i] > 0
            ]
        except Exception as e:
            log.warning("[BM25] Retrieve error: %s", e)
            return []


# ── FAISS ─────────────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Dense retrieval using sentence-transformers + FAISS.
    Builds index from any JSONL corpus; saves to disk for reuse.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._index: Any = None
        self._corpus: list[dict] = []

    def _lazy_load(self) -> None:
        if self._index is not None:
            return
        if not self.cfg.use_faiss:
            return
        index_path  = Path(self.cfg.faiss_index_path)
        corpus_path = Path(self.cfg.faiss_corpus_path)
        if not index_path.exists() or not corpus_path.exists():
            log.warning("[FAISS] Index not found at %s — building requires corpus.", index_path)
            self.cfg.use_faiss = False
            return
        try:
            import faiss
            self._index = faiss.read_index(str(index_path))
            with open(corpus_path, "rb") as f:
                self._corpus = pickle.load(f)
            log.info("[FAISS] Loaded index with %d vectors.", self._index.ntotal)
        except Exception as e:
            log.warning("[FAISS] Load error: %s", e)
            self.cfg.use_faiss = False

    def _embed(self, text: str):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.cfg.embedder_model)
        return model.encode([text], normalize_embeddings=True)

    async def retrieve(self, claim: str, _session: Any) -> list[Evidence]:
        if not self.cfg.use_faiss:
            return []
        self._lazy_load()
        if self._index is None:
            return []
        try:
            vec = await asyncio.to_thread(self._embed, claim)
            D, I = self._index.search(vec, self.cfg.top_k_per_source)
            ev = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._corpus):
                    continue
                doc = self._corpus[idx]
                ev.append(Evidence(
                    text=doc.get("text","")[:600], source="faiss",
                    title=doc.get("title",""), url=doc.get("url",""),
                    score=float(dist),
                ))
            return ev
        except Exception as e:
            log.warning("[FAISS] Retrieve error: %s", e)
            return []


# ── Hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Runs all enabled backends in parallel, merges, deduplicates,
    and returns the top-k highest-scored snippets.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backends = [
            WikipediaRetriever(cfg),
            TavilyRetriever(cfg),
            BM25Retriever(cfg),
            FAISSRetriever(cfg),
        ]

    async def retrieve(self, claim: str) -> list[Evidence]:
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [b.retrieve(claim, session) for b in self.backends]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[Evidence] = []
        for r in all_results:
            if isinstance(r, list):
                merged.extend(r)

        # Deduplicate by text similarity (simple prefix check)
        seen, deduped = set(), []
        for ev in sorted(merged, key=lambda e: -e.score):
            key = ev.text[:80].lower()
            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        return deduped[:self.cfg.top_k_final]


# ── Index builder (run once) ──────────────────────────────────────────────────

def build_faiss_index(
    corpus_jsonl: str,
    index_out: str,
    corpus_out: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> None:
    """
    Build a FAISS flat-L2 index from a JSONL corpus.
    Each line: {"text": "...", "title": "...", "url": "..."}

    Usage
    -----
    python -c "
    from truth_through_debate.retrieval.hybrid_retriever import build_faiss_index
    build_faiss_index('data/wiki_corpus.jsonl', 'data/faiss.index', 'data/faiss_corpus.pkl')
    "
    """
    import json, pickle, numpy as np, faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    corpus, texts = [], []
    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj)
            texts.append(obj.get("text","")[:400])

    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, index_out)
    with open(corpus_out, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Saved index ({index.ntotal} vectors) → {index_out}")
    print(f"Saved corpus ({len(corpus)} docs) → {corpus_out}")
