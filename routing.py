"""Query routing layer: BERT + LeetCode-RAG difficulty estimation + ELO router.

Pipeline:
    user query
      -> BERT embed (sentence-transformers MiniLM-L6-v2 by default)
      -> KNN against precomputed LeetCode embeddings (zerotrac/ratings.txt)
      -> weighted-avg neighbor ratings = predicted LeetCode rating
      -> linear map LC rating -> target model ELO
      -> pick the available GGUF whose ELO is closest to target

Seeds:
    Initial model ELOs come from LMSYS Chatbot Arena (Coding category)
    approximations, keyed by filename family patterns. Treat as priors
    only — Q4 quantization on a Pi typically shaves 50-100 ELO and votes
    will correct over time. The elo_state.json sidecar overrides any seed.

Vote loop:
    Thumbs up/down on a reply runs one standard ELO round between the
    model and the predicted target ELO (K = 32).

Fallback:
    If sentence-transformers isn't installed or the RAG index hasn't been
    built, Router falls back to a keyword heuristic so the system still
    works — routing just becomes coarser.
"""

import json
import os
import re

DEFAULT_ELO = 1100
K_FACTOR = 32

# LMSYS Chatbot Arena (Coding category) approximations, snapshot around
# late 2024 / early 2025. Patterns match against the lowercased GGUF
# filename. Caveats: Arena scores are full-precision cloud models — Q4_K_M
# on a Pi performs measurably worse; votes will adjust over time.
LMSYS_SEEDS = [
    (re.compile(r"qwen2\.5-coder-32b|qwen3-coder-30b", re.I), 1300),
    (re.compile(r"qwen2\.5-coder-14b", re.I),                 1240),
    (re.compile(r"qwen2\.5-coder-7b",  re.I),                 1185),
    (re.compile(r"qwen2\.5-coder-3b",  re.I),                 1115),
    (re.compile(r"qwen2\.5-coder-1\.5b", re.I),               1060),
    (re.compile(r"qwen2\.5-coder-0\.5b", re.I),                980),
    (re.compile(r"qwen2\.5-72b|qwen3-72b", re.I),             1260),
    (re.compile(r"qwen2\.5-32b",       re.I),                 1230),
    (re.compile(r"qwen2\.5-14b",       re.I),                 1190),
    (re.compile(r"qwen2\.5-7b",        re.I),                 1175),
    (re.compile(r"qwen2\.5-3b",        re.I),                 1095),
    (re.compile(r"qwen2\.5-1\.5b",     re.I),                 1045),
    (re.compile(r"llama-3\.3-70b|llama-3\.1-70b", re.I),      1250),
    (re.compile(r"llama-3\.1-8b|llama-3-8b",      re.I),      1165),
    (re.compile(r"llama-3\.2-3b",      re.I),                 1095),
    (re.compile(r"gemma-2-27b",        re.I),                 1210),
    (re.compile(r"gemma-2-9b",         re.I),                 1170),
    (re.compile(r"gemma-2-2b",         re.I),                 1110),
    (re.compile(r"codegemma-7b",       re.I),                 1115),
    (re.compile(r"codegemma-2b",       re.I),                 1015),
    (re.compile(r"phi-?3\.5-mini|phi-?3-mini", re.I),         1095),
    (re.compile(r"phi-?3-medium",      re.I),                 1130),
    (re.compile(r"phi-?4",             re.I),                 1200),
    (re.compile(r"deepseek-coder-v2-lite", re.I),             1180),
    (re.compile(r"deepseek-coder-(6\.7b|7b)", re.I),          1140),
    (re.compile(r"mistral-7b",         re.I),                 1060),
    (re.compile(r"tinyllama",          re.I),                  870),
]

# Coarse size-based fallback when no LMSYS pattern matches.
SIZE_FALLBACK = [
    (re.compile(r"(?<![0-9])(70b|72b)(?![a-z0-9])",          re.I), 1240),
    (re.compile(r"(?<![0-9])(30b|32b|33b|34b)(?![a-z0-9])",  re.I), 1210),
    (re.compile(r"(?<![0-9])(13b|14b)(?![a-z0-9])",          re.I), 1180),
    (re.compile(r"(?<![0-9])(7b|8b)(?![a-z0-9])",            re.I), 1130),
    (re.compile(r"(?<![0-9])(3b|4b)(?![a-z0-9])",            re.I), 1080),
    (re.compile(r"(?<![0-9])(1\.5b|1\.6b|1\.8b|2b|1\.1b)(?![a-z0-9])", re.I), 1020),
    (re.compile(r"(?<![0-9])(0\.5b|350m|125m|tiny)(?![a-z0-9])", re.I), 940),
]


def infer_elo_from_id(model_id: str) -> int:
    for pat, elo in LMSYS_SEEDS:
        if pat.search(model_id):
            return elo
    for pat, elo in SIZE_FALLBACK:
        if pat.search(model_id):
            return elo
    return DEFAULT_ELO


# ---------------------------------------------------------------------------
# ELO store
# ---------------------------------------------------------------------------

class EloStore:
    """Persistent ELO sidecar (JSON) keyed by model_id."""

    def __init__(self, path: str):
        self.path = path
        self._cache = self._load()

    def _load(self):
        if not os.path.isfile(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: int(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save(self):
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get(self, model_id: str) -> int:
        if model_id in self._cache:
            return int(self._cache[model_id])
        return infer_elo_from_id(model_id)

    def set(self, model_id: str, elo: int):
        self._cache[model_id] = int(elo)
        self._save()

    def all(self) -> dict:
        return dict(self._cache)


# ---------------------------------------------------------------------------
# LC rating -> target model ELO mapping
# ---------------------------------------------------------------------------

LC_RATING_FLOOR = 1100.0
LC_RATING_CEIL  = 3500.0
TARGET_ELO_FLOOR = 950
TARGET_ELO_CEIL  = 1450


def lc_rating_to_target_elo(lc_rating: float) -> int:
    """Linear map LC rating [1100..3500] -> target model ELO [950..1450]."""
    lc = max(LC_RATING_FLOOR, min(LC_RATING_CEIL, float(lc_rating)))
    frac = (lc - LC_RATING_FLOOR) / (LC_RATING_CEIL - LC_RATING_FLOOR)
    return int(round(TARGET_ELO_FLOOR + frac * (TARGET_ELO_CEIL - TARGET_ELO_FLOOR)))


# ---------------------------------------------------------------------------
# LeetCode RAG index
# ---------------------------------------------------------------------------

class LeetCodeRAG:
    """Loads embeddings + ratings produced by scripts/setup_rag.py."""

    def __init__(self, embeddings_path: str, meta_path: str):
        import numpy as np  # local — keep base startup light

        self._np = np
        emb = np.load(embeddings_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if emb.shape[0] != len(meta):
            raise ValueError(
                f"RAG index mismatch: {emb.shape[0]} embeddings vs "
                f"{len(meta)} meta rows"
            )
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms  # unit-normalised so cosine = dot
        self.meta = meta

    def knn_rating(self, query_vec, k: int = 5):
        np = self._np
        q = query_vec.astype("float32").reshape(-1)
        qn = float((q * q).sum()) ** 0.5
        if qn == 0:
            return float(LC_RATING_FLOOR), []
        q = q / qn
        sims = self.embeddings @ q
        k = max(1, min(k, sims.shape[0]))
        if k < sims.shape[0]:
            idx = np.argpartition(-sims, kth=k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
        else:
            idx = np.argsort(-sims)
        weights = np.clip(sims[idx], 0.0, 1.0) + 1e-3
        ratings = np.array([float(self.meta[i]["rating"]) for i in idx])
        avg = float((weights * ratings).sum() / weights.sum())
        neighbors = [
            {"title": self.meta[i]["title"],
             "rating": float(self.meta[i]["rating"]),
             "similarity": float(sims[i])}
            for i in idx
        ]
        return avg, neighbors


# ---------------------------------------------------------------------------
# BERT embedder
# ---------------------------------------------------------------------------

class BertEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self.model.encode([text], normalize_embeddings=False)[0]


# ---------------------------------------------------------------------------
# Keyword fallback (used when BERT/RAG aren't available)
# ---------------------------------------------------------------------------

HARD_HINTS = [
    "concurrent", "thread safe", "thread-safe", "lock-free", "lockless",
    "race condition", "deadlock", "atomic", "memory order", "mutex",
    "kernel module", "compiler internals", "register allocation",
    "cuda", "tensor", "vectorize", "simd", "avx", "neon", "intrinsic",
    "assembly", "inline asm", "x86_64", "arm64",
    "monad", "category theory", "dependent type",
    "amortized analysis", "asymptotic proof", "np-hard", "np-complete",
    "system design", "distributed system", "consensus", "raft", "paxos",
    "garbage collector", "jit compiler",
    "leetcode hard", "interview question", "advanced algorithm",
]
EASY_HINTS = [
    "hello world", "print", "syntax of", "what is a",
    "for loop", "while loop", "if statement",
    "string concatenation", "basic", "beginner",
]


def keyword_estimate_lc_rating(message: str):
    msg = (message or "").lower().strip()
    words = msg.split()
    wc = len(words)
    if any(h in msg for h in HARD_HINTS):
        return 2400.0, "keyword:hard-hint"
    if wc > 80 or len(msg) > 500:
        return 2200.0, "keyword:long-prompt"
    if wc <= 6:
        return 1200.0, "keyword:very-short"
    if wc <= 12 and any(h in msg for h in EASY_HINTS):
        return 1300.0, "keyword:easy-hint"
    return 1800.0, "keyword:default-medium"


# ---------------------------------------------------------------------------
# Router front-end
# ---------------------------------------------------------------------------

class Router:
    """Estimate query difficulty, pick a model whose ELO best matches."""

    def __init__(self,
                 embeddings_path: str | None,
                 meta_path: str | None,
                 bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings_path = embeddings_path
        self.meta_path = meta_path
        self.bert_model_name = bert_model_name
        self._rag = None
        self._embedder = None
        self._tried_load = False
        self._load_error: str | None = None

    def _ensure_loaded(self):
        if self._tried_load:
            return
        self._tried_load = True
        if not self.embeddings_path or not self.meta_path:
            self._load_error = "RAG paths not configured"
            return
        if not (os.path.isfile(self.embeddings_path) and os.path.isfile(self.meta_path)):
            self._load_error = "RAG index not built — run scripts/setup_rag.py"
            return
        try:
            self._rag = LeetCodeRAG(self.embeddings_path, self.meta_path)
            self._embedder = BertEmbedder(self.bert_model_name)
        except Exception as e:
            self._load_error = f"RAG load failed: {e}"
            self._rag = None
            self._embedder = None

    @property
    def using_bert(self) -> bool:
        self._ensure_loaded()
        return self._rag is not None and self._embedder is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def estimate(self, message: str) -> dict:
        self._ensure_loaded()
        if self._rag is not None and self._embedder is not None:
            try:
                vec = self._embedder.encode(message)
                rating, neighbors = self._rag.knn_rating(vec, k=5)
                return {
                    "predicted_lc_rating": round(rating, 1),
                    "target_elo": lc_rating_to_target_elo(rating),
                    "source": "bert_rag",
                    "neighbors": neighbors,
                }
            except Exception as e:
                self._load_error = f"BERT/RAG inference failed: {e}"
        rating, src = keyword_estimate_lc_rating(message)
        return {
            "predicted_lc_rating": rating,
            "target_elo": lc_rating_to_target_elo(rating),
            "source": src,
            "rag_unavailable": self._load_error,
            "neighbors": [],
        }

    def route(self, message: str, available_ids: list, elo_store: EloStore) -> dict:
        est = self.estimate(message)
        if not available_ids:
            est["chosen_model_id"] = None
            est["model_elo"] = None
            return est
        target = est["target_elo"]
        ranked = sorted(
            available_ids,
            key=lambda mid: (abs(elo_store.get(mid) - target), elo_store.get(mid)),
        )
        chosen = ranked[0]
        est["chosen_model_id"] = chosen
        est["model_elo"] = elo_store.get(chosen)
        return est


def update_elo(elo_store: EloStore, model_id: str, target_elo: int, vote: str) -> int:
    """Standard ELO update; vote='up' = score 1.0, 'down' = 0.0. K=32."""
    score = 1.0 if vote == "up" else 0.0
    current = elo_store.get(model_id)
    expected = 1.0 / (1.0 + 10 ** ((target_elo - current) / 400.0))
    new_elo = int(round(current + K_FACTOR * (score - expected)))
    elo_store.set(model_id, new_elo)
    return new_elo
