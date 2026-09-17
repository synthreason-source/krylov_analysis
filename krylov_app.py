from __future__ import annotations

"""
Krylov Complexity Detector based on n-gram language model kernel.

What's new:
  - Krylov subspace construction via Lanczos algorithm on token transition operators
  - Lanczos coefficients {a_n, b_n} capture the "dynamics" of sequence evolution
  - Krylov complexity K(t) = Σ n |φ_n(t)|² measures operator spreading
  - Anomaly detection via deviation from expected Lanczos coefficient growth
  - Krylov entropy as complementary disorder measure
  - Change-point detection in sequences via sliding-window K-complexity

The detector treats the trained n-gram model as defining a Liouvillian/operator
that governs token transitions. By running Lanczos recursion on this operator
starting from different seed tokens/contexts, we get:
  1. Lanczos coefficients that characterize the "dynamical structure"
  2. Krylov complexity growth curves that detect structural breaks
  3. Anomaly scores based on coefficient pattern deviations

Still pure Python + numpy. No torch. Gradio UI updated with detector controls.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import gradio as gr

MODEL_PATH = "model.json"
DEFAULT_CORPUS_FILE = "corpus.txt"

MAX_NEW_TOKENS = 800
TEMPERATURE = 0.8
TOP_K = 20
MIN_COUNT = 1
INSTRUCTION_WEIGHT = 1.5

PARADIGM_STEPS = 10
PARADIGM_WEIGHT = 1.2

LEXICAL_WEIGHT = 0.15
VECTOR_WEIGHT = 0.15

IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}

# Krylov detector parameters
KRYLOV_MAX_ITER = 50
KRYLOV_WINDOW_SIZE = 5
KRYLOV_STRIDE = 1
ANOMALY_THRESHOLD = 2.0  # standard deviations

SCORECARD_PATH = "anomaly_scorecard.png"


# ------------------- Krylov complexity core ------------------

@dataclass
class KrylovResult:
    """Results from Krylov complexity analysis."""
    lanczos_a: List[float]  # diagonal coefficients
    lanczos_b: List[float]  # off-diagonal coefficients
    complexity_curve: List[float]  # K(t) over "time" steps
    krylov_entropy: float  # entropy of |φ_n|² distribution
    anomaly_score: float  # deviation from expected b_n pattern
    basis_size: int  # actual Krylov dimension reached


def _shannon_entropy_bits(v: np.ndarray) -> float:
    """Shannon entropy in bits for a probability distribution."""
    magnitudes = np.abs(v)
    total = magnitudes.sum()
    if total <= 0:
        return 0.0
    probabilities = magnitudes[magnitudes > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _lanczos_recursion(
    operator_matrix: np.ndarray,
    initial_vector: np.ndarray,
    max_iter: int = 50,
    tolerance: float = 1e-10
) -> Tuple[List[float], List[float], int]:
    """
    Lanczos algorithm for tridiagonalization.

    Given a symmetric operator L and starting vector v_0, produces:
      - a_n = ⟨v_n | L | v_n⟩ (diagonal)
      - b_n = ‖w_n‖ where w_n = L v_n - a_n v_n - b_{n-1} v_{n-1}

    The b_n coefficients control spreading in Krylov space.
    Returns (a_coeffs, b_coeffs, actual_iterations).
    """
    n = operator_matrix.shape[0]
    if n == 0:
        return [], [], 0

    v_prev = np.zeros(n, dtype=np.float64)
    v_curr = initial_vector / (np.linalg.norm(initial_vector) + 1e-12)

    a_coeffs = []
    b_coeffs = []
    b_prev = 0.0

    for iteration in range(max_iter):
        w = operator_matrix @ v_curr

        a_n = float(np.dot(v_curr, w))
        a_coeffs.append(a_n)

        w = w - a_n * v_curr
        if iteration > 0:
            w = w - b_prev * v_prev

        b_n = np.linalg.norm(w)

        if b_n < tolerance or iteration == max_iter - 1:
            b_coeffs.append(0.0)
            return a_coeffs, b_coeffs, iteration + 1

        b_coeffs.append(b_n)
        b_prev = b_n

        v_prev = v_curr.copy()
        v_curr = w / (b_n + 1e-12)

    return a_coeffs, b_coeffs, max_iter


def _build_transition_operator(model: "NGramModel", context_tokens: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Build a finite-dimensional transition operator from n-gram statistics.
    """
    if not model.finalized:
        model.finalize()

    if context_tokens:
        ctx = context_tokens[-2:] if len(context_tokens) >= 2 else context_tokens
        base_dist = model.backoff_distribution(ctx[-1], ctx[-2] if len(ctx) > 1 else None)
    else:
        base_dist = model.normalize(model.unigram)

    sorted_tokens = sorted(base_dist.items(), key=lambda x: x[1], reverse=True)
    max_dim = min(64, len(sorted_tokens))
    active_tokens = [t for t, _ in sorted_tokens[:max_dim]]

    if len(active_tokens) < 2:
        return np.zeros((1, 1)), active_tokens or ["<unk>"], {t: i for i, t in enumerate(active_tokens or ["<unk>"])}

    token_to_idx = {t: i for i, t in enumerate(active_tokens)}
    n = len(active_tokens)

    L = np.zeros((n, n), dtype=np.float64)

    for i, token_i in enumerate(active_tokens):
        next_dist = model.backoff_distribution(token_i, context_tokens[-1] if context_tokens else None)

        for j, token_j in enumerate(active_tokens):
            p_ij = next_dist.get(token_j, 1e-12)
            p_baseline = base_dist.get(token_j, 1e-12)

            if p_ij > 1e-12 and p_baseline > 1e-12:
                L[i, j] = math.log(p_ij / p_baseline)
            else:
                L[i, j] = 0.0

    L = (L + L.T) / 2.0

    return L, active_tokens, token_to_idx


def _compute_krylov_complexity(
    model: "NGramModel",
    seed_context: List[str],
    max_iter: int = KRYLOV_MAX_ITER,
    time_steps: int = 20
) -> KrylovResult:
    """
    Compute Krylov complexity for sequence evolution from a seed context.
    """
    if not model.finalized:
        model.finalize()

    L, tokens, token_to_idx = _build_transition_operator(model, seed_context)
    n_dim = L.shape[0]

    if n_dim < 2:
        return KrylovResult(
            lanczos_a=[0.0],
            lanczos_b=[0.0],
            complexity_curve=[0.0],
            krylov_entropy=0.0,
            anomaly_score=0.0,
            basis_size=1
        )

    if seed_context and seed_context[-1] in token_to_idx:
        v0 = np.zeros(n_dim, dtype=np.float64)
        v0[token_to_idx[seed_context[-1]]] = 1.0
    else:
        v0 = np.ones(n_dim, dtype=np.float64) / math.sqrt(n_dim)

    a_coeffs, b_coeffs, actual_iter = _lanczos_recursion(L, v0, max_iter=max_iter)

    complexity_curve = []
    for t_step in range(time_steps):
        t = t_step * 0.5

        phi_sq = []
        for n_idx in range(len(b_coeffs) - 1):
            b_effective = np.mean(b_coeffs[:min(n_idx+1, len(b_coeffs)-1)]) if b_coeffs else 1.0
            if n_idx == 0:
                prob = max(0.0, 1.0 - b_effective * t)
            else:
                prob = (b_effective * t) ** (2 * n_idx) / math.factorial(n_idx + 1) ** 2
                prob = min(prob, 1.0)
            phi_sq.append(prob)

        total = sum(phi_sq) + 1e-12
        phi_sq = [p / total for p in phi_sq]

        K_t = sum(n * p for n, p in enumerate(phi_sq))
        complexity_curve.append(K_t)

    phi_final = []
    for n_idx in range(len(b_coeffs) - 1):
        b_effective = np.mean(b_coeffs[:min(n_idx+1, len(b_coeffs)-1)]) if b_coeffs else 1.0
        t = (time_steps - 1) * 0.5
        if n_idx == 0:
            prob = max(0.0, 1.0 - b_effective * t)
        else:
            prob = (b_effective * t) ** (2 * n_idx) / math.factorial(n_idx + 1) ** 2
            prob = min(prob, 1.0)
        phi_final.append(prob)

    total = sum(phi_final) + 1e-12
    phi_final = [p / total for p in phi_final]
    krylov_entropy = _shannon_entropy_bits(np.array(phi_final))

    if len(b_coeffs) > 3:
        b_nonzero = [b for b in b_coeffs[:-1] if b > 1e-6]
        if len(b_nonzero) > 2:
            x = np.arange(len(b_nonzero))
            y = np.array(b_nonzero)
            coeffs = np.polyfit(x, y, 1)
            trend = coeffs[0] * x + coeffs[1]
            residuals = y - trend
            std_residual = np.std(residuals)
            mean_b = np.mean(b_nonzero)
            anomaly_score = std_residual / (mean_b + 1e-6)
        else:
            anomaly_score = 0.0
    else:
        anomaly_score = 0.0

    return KrylovResult(
        lanczos_a=a_coeffs,
        lanczos_b=b_coeffs,
        complexity_curve=complexity_curve,
        krylov_entropy=krylov_entropy,
        anomaly_score=anomaly_score,
        basis_size=actual_iter
    )


def _sliding_window_krylov(
    model: "NGramModel",
    token_sequence: List[str],
    window_size: int = KRYLOV_WINDOW_SIZE,
    stride: int = KRYLOV_STRIDE
) -> List[Tuple[int, KrylovResult]]:
    """
    Compute Krylov complexity in sliding windows across a sequence.
    """
    results = []

    for start in range(0, len(token_sequence) - window_size + 1, stride):
        window = token_sequence[start:start + window_size]
        krylov_result = _compute_krylov_complexity(model, window)
        results.append((start, krylov_result))

    return results


def _detect_anomalies(
    krylov_results: List[Tuple[int, KrylovResult]],
    threshold: float = ANOMALY_THRESHOLD
) -> List[Tuple[int, float]]:
    """
    Detect anomalous windows based on Krylov metrics.
    """
    if not krylov_results:
        return []

    anomaly_scores = [r[1].anomaly_score for r in krylov_results]
    entropies = [r[1].krylov_entropy for r in krylov_results]
    basis_sizes = [r[1].basis_size for r in krylov_results]

    if not anomaly_scores:
        return []

    median_anomaly = np.median(anomaly_scores)
    std_anomaly = np.std(anomaly_scores) + 1e-6
    median_entropy = np.median(entropies)
    std_entropy = np.std(entropies) + 1e-6

    anomalies = []
    for idx, (start_pos, result) in enumerate(krylov_results):
        z_anomaly = (result.anomaly_score - median_anomaly) / std_anomaly
        z_entropy = abs(result.krylov_entropy - median_entropy) / std_entropy

        z_basis = 0.0
        if median_basis := np.median(basis_sizes):
            if result.basis_size < 0.5 * median_basis:
                z_basis = 2.0

        combined_score = max(z_anomaly, z_entropy, z_basis)

        if combined_score > threshold:
            anomalies.append((start_pos, combined_score))

    return anomalies


# --------------------------- shared helpers --------------------------

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in text.split(".") if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-12) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a < eps or norm_b < eps:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a) - IGNORED_TOKENS, set(b) - IGNORED_TOKENS
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def strip_structural_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in IGNORED_TOKENS]


def _dense_matrix_from_vectors(vectors: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """Stack sparse dict-vectors into one dense matrix over their shared key space."""
    keys = sorted({k for v in vectors for k in v})
    key_index = {k: i for i, k in enumerate(keys)}
    matrix = np.zeros((len(vectors), len(keys)), dtype=np.float64)
    for row, vector in enumerate(vectors):
        for key, weight in vector.items():
            matrix[row, key_index[key]] = weight
    return matrix, keys


def _dominant_eigenvector(matrix: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Power iteration for the leading eigenvector of a symmetric matrix."""
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    vector = np.ones(n, dtype=np.float64) / math.sqrt(n)
    for _ in range(iterations):
        next_vector = matrix @ vector
        norm = np.linalg.norm(next_vector)
        if norm < 1e-12:
            return vector
        next_vector = next_vector / norm
        if np.allclose(next_vector, vector, atol=1e-10) or np.allclose(next_vector, -vector, atol=1e-10):
            return next_vector
        vector = next_vector
    return vector


# --------------------------- corpus search ----------------------------

@dataclass
class CorpusReference:
    sentence: str
    tokens: List[str]
    vector: Dict[str, float]
    frequency: int = 1


@dataclass
class Candidate:
    sentence: str
    symbolic_overlap: float
    vector_similarity: float
    frequency: int
    score: float
    rank: int = 0


class CorpusSearch:
    def __init__(self, lexical_weight=LEXICAL_WEIGHT, vector_weight=VECTOR_WEIGHT):
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(s.lower() for s in sentences)
        self.references = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            bow = bag_of_words(tokens)
            self.references.append(
                CorpusReference(
                    sentence=sentence,
                    tokens=tokens,
                    vector={t: float(c) for t, c in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {t: float(c) for t, c in bag_of_words(prompt_tokens).items()}

        candidates = []
        for ref in self.references:
            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            vec_sim = cosine_similarity(prompt_vector, ref.vector)
            score = self.lexical_weight * symbolic + self.vector_weight * vec_sim
            candidates.append(
                Candidate(ref.sentence, symbolic, vec_sim, ref.frequency, score)
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates


# --------------------------- the kernel --------------------------------

@dataclass
class NGramModel:
    """Trigram-backoff language model with Krylov complexity detector."""

    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    # ---- training ----

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue
            self._add_sequence(["<bos>", "<bos>", *words, self.eos_token])

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts: Dict[str, Counter] = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {c: n / total for c, n in counts.items()}

        self.finalized = True

    # ---- prompt kernel (paradigm for the conversation) ----

    def prompt_kernel_vector(self, prompt: str) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        tokens = [t for t in tokenize(prompt) if t not in IGNORED_TOKENS]
        tokens = [t for t in tokens if self.lexical_vectors.get(t)]
        if not tokens:
            return {}

        vectors = [self.lexical_vectors[t] for t in tokens]
        matrix, keys = _dense_matrix_from_vectors(vectors)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normalized = matrix / norms

        kernel = normalized @ normalized.T
        weights = np.abs(_dominant_eigenvector(kernel))
        total = weights.sum()
        if total < 1e-12:
            return {}
        weights = weights / total

        paradigm = weights @ matrix
        return {key: float(w) for key, w in zip(keys, paradigm) if abs(w) > 1e-9}

    # ---- scoring / sampling ----

    def backoff_distribution(self, previous: str, previous_previous: Optional[str]) -> Dict[str, float]:
        if previous_previous is not None:
            counts = self.trigram.get(f"{previous_previous}\t{previous}")
            if counts:
                return self.normalize(counts)
        counts = self.bigram.get(previous)
        if counts:
            return self.normalize(counts)
        return self.normalize(self.unigram)

    @staticmethod
    def normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def resolve_context(self, prompt: str) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)
        if not tokens:
            return "<bos>", None
        previous = tokens[-1]
        previous_previous = tokens[-2] if len(tokens) >= 2 else None
        return previous, previous_previous

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_context(prompt)
        base = self.backoff_distribution(previous, previous_previous)
        if not base:
            return {}

        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]

        scores: Dict[str, float] = {}
        for token in candidates:
            score = safe_log(base[token])

            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(instruction_vector, self.lexical_vectors.get(token, {}))
                score += instruction_weight * likeness

            scores[token] = score

        return scores

    def probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> Dict[str, float]:
        scores = self.score_next_token(
            prompt, candidate_limit, instruction_vector, instruction_weight
        )
        if not scores:
            return {}

        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> str:
        probs = self.probabilities(
            prompt, temperature, max(top_k, 1), instruction_vector, instruction_weight
        )
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        paradigm_steps: int = 0,
        paradigm_weight: float = PARADIGM_WEIGHT,
    ) -> str:
        generated = tokenize(prompt)

        paradigm_vector = self.prompt_kernel_vector(prompt) if paradigm_steps > 0 else None

        for step in range(max_new_tokens):
            if paradigm_vector and step < paradigm_steps:
                active_vector, active_weight = paradigm_vector, paradigm_weight
            else:
                active_vector, active_weight = instruction_vector, instruction_weight

            token = self.sample_next(
                " ".join(generated), temperature, top_k, active_vector, active_weight
            )
            generated.append(token)

        return self.detokenize(strip_structural_tokens(generated))

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data.get("eos_token", "<eos>"),
            unk_token=data.get("unk_token", "<unk>"),
            min_count=data.get("min_count", MIN_COUNT),
        )
        model.unigram = Counter(data.get("unigram", {}))
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data.get("bigram", {}).items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data.get("trigram", {}).items()})
        model.lexical_vectors = data.get("lexical_vectors", {})
        model.vocabulary = data.get("vocabulary", [])
        model.finalized = data.get("finalized", False)
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------- globals & UI wiring ---------------------------

TEXT_MODEL: NGramModel | None = None
CORPUS_SEARCH: CorpusSearch | None = None
CORPUS_TEXT_CACHE: str | None = None


def load_text_model() -> NGramModel:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Upload a corpus and click 'Train model' first.")
    model = NGramModel.load_json(model_path)
    if not model.finalized:
        model.finalize()
    return model


def load_corpus_search(corpus_text: str) -> CorpusSearch:
    search = CorpusSearch()
    search.build_index(corpus_text)
    return search


def reload_globals():
    global TEXT_MODEL, CORPUS_SEARCH, CORPUS_TEXT_CACHE
    TEXT_MODEL = load_text_model()

    if CORPUS_TEXT_CACHE:
        corpus_text = CORPUS_TEXT_CACHE
    else:
        default_path = Path(DEFAULT_CORPUS_FILE)
        corpus_text = default_path.read_text(encoding="utf-8", errors="replace") if default_path.exists() else ""
        CORPUS_TEXT_CACHE = corpus_text

    CORPUS_SEARCH = load_corpus_search(corpus_text)


def format_matches(prompt: str, limit: int = 5) -> str:
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus file loaded."
    candidates = CORPUS_SEARCH.analyze(prompt, limit=limit)
    if not candidates:
        return "No corpus matches found."
    return "\n".join(
        f"{c.rank}. {c.sentence} (score={c.score:.3f}, overlap={c.symbolic_overlap:.3f}, vector={c.vector_similarity:.3f})"
        for c in candidates
    )


def train_model_from_file(corpus_file):
    global CORPUS_TEXT_CACHE
    if corpus_file is None:
        return "No corpus file uploaded.", "", ""

    path = Path(corpus_file.name) if hasattr(corpus_file, "name") else Path(corpus_file)
    if not path.exists():
        return f"Corpus file not found: {path}", "", ""

    corpus_text = path.read_text(encoding="utf-8", errors="replace")
    CORPUS_TEXT_CACHE = corpus_text

    model = NGramModel()
    model.ingest_text(corpus_text)
    model.finalize()
    model.save_json(Path(MODEL_PATH))

    reload_globals()

    sample_lines = "\n".join(corpus_text.splitlines()[:5])
    summary = (
        f"Vocabulary: {len(model.vocabulary)}\n"
        f"Unigrams: {len(model.unigram)}\n"
        f"Bigram contexts: {len(model.bigram)}\n"
        f"Trigram contexts: {len(model.trigram)}"
    )
    return f"Model trained and saved to {MODEL_PATH}.", sample_lines, summary


def generate_from_prompt(user_prompt: str, paradigm_steps: int):
    if TEXT_MODEL is None or CORPUS_SEARCH is None:
        return "Model not loaded. Upload a corpus and click 'Train model' first.", ""

    prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else "<bos>"

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        paradigm_steps=int(paradigm_steps),
        paradigm_weight=PARADIGM_WEIGHT,
    )
    corpus_matches = format_matches(prompt, limit=5) if prompt != "<bos>" else "No prompt provided for corpus search."

    return corpus_matches, generated


# ------------------------- Krylov detector UI functions -------------------------

def analyze_krylov_single(prompt: str, max_iter: int) -> str:
    """Analyze Krylov complexity for a single prompt."""
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(prompt)
    if not tokens:
        return "Empty prompt."

    result = _compute_krylov_complexity(TEXT_MODEL, tokens, max_iter=max_iter)

    output_lines = [
        f"Krylov Analysis for: '{prompt[:50]}...' ",
        "",
        f"Basis size: {result.basis_size}",
        f"Lanczos a (diagonal): {[round(a, 4) for a in result.lanczos_a[:10]]}{'...' if len(result.lanczos_a) > 10 else ''}",
        f"Lanczos b (off-diagonal): {[round(b, 4) for b in result.lanczos_b[:10]]}{'...' if len(result.lanczos_b) > 10 else ''}",
        "",
        f"Krylov entropy: {result.krylov_entropy:.4f} bits",
        f"Anomaly score: {result.anomaly_score:.4f}",
        "",
        "Complexity curve K(t):",
    ]

    for t, k_val in enumerate(result.complexity_curve):
        output_lines.append(f"  t={t}: K = {k_val:.4f}")

    return "\n".join(output_lines)


def analyze_krylov_sequence(sequence_text: str, window_size: int, stride: int, threshold: float) -> str:
    """Analyze Krylov complexity across a token sequence with sliding windows."""
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(sequence_text)
    if len(tokens) < window_size:
        return f"Sequence too short ({len(tokens)} tokens). Need at least {window_size}."

    window_results = _sliding_window_krylov(TEXT_MODEL, tokens, window_size=window_size, stride=stride)

    if not window_results:
        return "No windows analyzed."

    anomalies = _detect_anomalies(window_results, threshold=threshold)

    output_lines = [
        f"Krylov Sequence Analysis",
        f"Sequence length: {len(tokens)} tokens",
        f"Window size: {window_size}, stride: {stride}",
        "",
        "Window-by-window results:",
    ]

    for start_pos, result in window_results:
        window_tokens = tokens[start_pos:start_pos + window_size]
        window_text = " ".join(window_tokens[:10]) + ("..." if len(window_tokens) > 10 else "")

        anomaly_flag = " [ANOMALY]" if any(abs(start_pos - a[0]) < stride for a in anomalies) else ""
        output_lines.append(
            f"\n  Window {start_pos}-{start_pos + window_size}: {window_text}"
            f"\n    K={result.complexity_curve[-1]:.3f}, "
            f"S={result.krylov_entropy:.3f}, "
            f"anomaly={result.anomaly_score:.3f}{anomaly_flag}"
        )

    if anomalies:
        output_lines.append("")
        output_lines.append(f"Detected {len(anomalies)} anomalies at positions:")
        for pos, score in anomalies:
            anomaly_tokens = tokens[pos:pos + min(10, len(tokens) - pos)]
            anomaly_text = " ".join(anomaly_tokens)
            output_lines.append(f"  Position {pos}: {anomaly_text} (score = {score:.3f})")
    else:
        output_lines.append("")
        output_lines.append("No anomalies detected.")

    return "\n".join(output_lines)


def detect_change_points(sequence_text: str, window_size: int) -> str:
    """Detect structural change points in sequence via Krylov complexity."""
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(sequence_text)
    if len(tokens) < 2 * window_size:
        return f"Sequence too short for change-point detection."

    stride = max(1, window_size // 4)
    window_results = _sliding_window_krylov(TEXT_MODEL, tokens, window_size=window_size, stride=stride)

    if len(window_results) < 3:
        return "Not enough windows for change-point detection."

    change_points = []
    prev_result = None

    for start_pos, result in window_results:
        if prev_result is not None:
            delta_k = abs(result.complexity_curve[-1] - prev_result.complexity_curve[-1])
            delta_s = abs(result.krylov_entropy - prev_result.krylov_entropy)
            delta_anomaly = abs(result.anomaly_score - prev_result.anomaly_score)

            if delta_k > 0.5 or delta_s > 0.3 or delta_anomaly > 1.0:
                change_points.append((start_pos, delta_k, delta_s, delta_anomaly))

        prev_result = result

    output_lines = [
        f"Change-Point Detection Results",
        f"Sequence: {len(tokens)} tokens, window size: {window_size}",
        "",
    ]

    if change_points:
        output_lines.append(f"Found {len(change_points)} potential change points:")
        for pos, dk, ds, da in change_points:
            context_tokens = tokens[max(0, pos-2):pos + min(8, len(tokens) - pos)]
            context_text = " ".join(context_tokens)
            output_lines.append(
                f"  Position {pos}: ...{context_text}..."
                f"\n    ΔK={dk:.3f}, ΔS={ds:.3f}, Δanomaly={da:.3f}"
            )
    else:
        output_lines.append("No significant change points detected.")
        output_lines.append("")
        output_lines.append("Sequence appears structurally homogeneous by Krylov metrics.")

    return "\n".join(output_lines)


# ------------------------- Anomaly scorecard PNG --------------------------
#
# Renders the *full* scanned text with every anomalous window's words
# wrapped in [ ], plus a small per-window score table underneath, as one
# PNG "scorecard" image the user can save or share.

def _bracket_anomalous_windows(
    tokens: List[str],
    window_size: int,
    anomaly_positions: Iterable[int],
) -> str:
    """Wrap every token covered by an anomalous window in [ ]. Adjacent or
    overlapping anomalous windows are merged into a single bracket run so
    the output doesn't read as "[word] [word] [word]".
    """
    covered = [False] * len(tokens)
    for pos in anomaly_positions:
        for i in range(pos, min(pos + window_size, len(tokens))):
            covered[i] = True

    pieces: List[str] = []
    i = 0
    while i < len(tokens):
        if covered[i]:
            j = i
            run: List[str] = []
            while j < len(tokens) and covered[j]:
                run.append(tokens[j])
                j += 1
            pieces.append("[" + " ".join(run) + "]")
            i = j
        else:
            pieces.append(tokens[i])
            i += 1

    return " ".join(pieces)


def _draw_scorecard_png(
    annotated_text: str,
    anomaly_count: int,
    window_count: int,
    threshold: float,
    output_path: str,
) -> None:
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrapped = textwrap.fill(annotated_text, width=78)
    # Size the figure to the text itself so nothing gets cut off: estimate
    # height from the number of wrapped lines instead of using a fixed
    # canvas that a longer sequence could overflow.
    line_count = max(wrapped.count("\n") + 1, 1)
    fig_height = 1.3 + line_count * 0.28

    fig, ax_text = plt.subplots(figsize=(11, fig_height))
    fig.suptitle("Krylov Anomaly Scorecard", fontsize=17, fontweight="bold")

    ax_text.axis("off")
    ax_text.set_title(
        f"Scanned text  —  {anomaly_count} of {window_count} windows flagged  "
        f"(threshold={threshold})  —  [bracketed] = anomalous",
        fontsize=11, loc="left", color="#333333",
    )
    ax_text.text(
        0.0, 1.0, wrapped, va="top", ha="left",
        fontsize=10.5, family="monospace", transform=ax_text.transAxes,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_anomaly_scorecard(
    sequence_text: str,
    window_size: int,
    stride: int,
    threshold: float,
) -> Tuple[Optional[str], str]:
    """Run the same sliding-window Krylov anomaly scan as the text tab, then
    render it as one PNG: just the full text with anomalous windows
    bracketed in [ ]. Returns (png_path_or_None, message).
    """
    if TEXT_MODEL is None:
        return None, "Model not loaded."

    tokens = tokenize(sequence_text)
    if len(tokens) < window_size:
        return None, f"Sequence too short ({len(tokens)} tokens). Need at least {window_size}."

    window_results = _sliding_window_krylov(TEXT_MODEL, tokens, window_size=window_size, stride=stride)
    if not window_results:
        return None, "No windows analyzed."

    anomalies = _detect_anomalies(window_results, threshold=threshold)
    anomaly_lookup = dict(anomalies)  # start_pos -> combined score

    annotated_text = _bracket_anomalous_windows(tokens, window_size, anomaly_lookup.keys())

    output_path = str(Path(SCORECARD_PATH).resolve())
    _draw_scorecard_png(
        annotated_text,
        anomaly_count=len(anomaly_lookup),
        window_count=len(window_results),
        threshold=threshold,
        output_path=output_path,
    )

    message = (
        f"{len(anomaly_lookup)} of {len(window_results)} windows flagged as anomalous "
        f"(threshold={threshold}). Scorecard saved."
    )
    return output_path, message


# ------------------------- Gradio UI ---------------------------

with gr.Blocks(title="Krylov Detector") as demo:
    gr.Markdown(
        """
# Krylov Complexity Detector

N-gram language model + Krylov subspace analysis for sequence anomaly detection.

**Workflow:**
1. Upload corpus → Train model
2. Use Krylov analysis tabs to detect anomalies, change-points, and structural breaks
3. Lanczos coefficients {a_n, b_n} characterize the "dynamics" of token transitions
4. Krylov complexity K(t) measures operator spreading - sudden changes indicate anomalies
"""
    )

    gr.Markdown("## 1. Corpus & Training")
    with gr.Row():
        with gr.Column(scale=1):
            corpus_file_input = gr.File(label="Corpus file (any type)", file_types=["file"])
            train_button = gr.Button("Train model", variant="primary")
        with gr.Column(scale=2):
            train_status = gr.Textbox(label="Training status", lines=2)
            corpus_sample = gr.Textbox(label="Corpus sample (first 5 lines)", lines=5)
            model_summary = gr.Textbox(label="Model summary", lines=6)

    train_button.click(
        fn=train_model_from_file,
        inputs=[corpus_file_input],
        outputs=[train_status, corpus_sample, model_summary],
    )

    with gr.Tab("Single Prompt Analysis"):
        gr.Markdown(
            """
### Krylov Analysis for Single Prompt

Computes Lanczos coefficients and complexity curve for one prompt's token dynamics.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                krylov_prompt = gr.Textbox(label="Prompt to analyze", lines=3)
                krylov_max_iter = gr.Slider(
                    minimum=5, maximum=100, value=30, step=1,
                    label="Max Lanczos iterations"
                )
                krylov_analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                krylov_output = gr.Textbox(label="Krylov analysis results", lines=15)

        krylov_analyze_btn.click(
            fn=analyze_krylov_single,
            inputs=[krylov_prompt, krylov_max_iter],
            outputs=[krylov_output],
        )

    with gr.Tab("Sequence Anomaly Detection"):
        gr.Markdown(
            """
### Sliding-Window Anomaly Detection

Scans a sequence with sliding windows, computing Krylov complexity for each.
Anomalies flagged where Lanczos b_n pattern deviates or complexity jumps.
Use **Generate scorecard PNG** to export the flagged windows, bracketed in
[ ] amongst the rest of the text, as one image.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                anomaly_sequence = gr.Textbox(label="Sequence to scan", lines=5)
                anomaly_window = gr.Slider(
                    minimum=5, maximum=50, value=5, step=1,
                    label="Window size"
                )
                anomaly_stride = gr.Slider(
                    minimum=1, maximum=20, value=1, step=1,
                    label="Stride"
                )
                anomaly_threshold = gr.Slider(
                    minimum=0.5, maximum=5.0, value=2.0, step=0.1,
                    label="Anomaly threshold (std devs)"
                )
                anomaly_detect_btn = gr.Button("Detect anomalies", variant="primary")
                scorecard_btn = gr.Button("Generate scorecard PNG", variant="secondary")
            with gr.Column(scale=2):
                anomaly_output = gr.Textbox(label="Anomaly detection results", lines=15)
                scorecard_image = gr.Image(label="Anomaly scorecard", type="filepath")
                scorecard_status = gr.Textbox(label="Scorecard status", lines=2)

        anomaly_detect_btn.click(
            fn=analyze_krylov_sequence,
            inputs=[anomaly_sequence, anomaly_window, anomaly_stride, anomaly_threshold],
            outputs=[anomaly_output],
        )

        scorecard_btn.click(
            fn=generate_anomaly_scorecard,
            inputs=[anomaly_sequence, anomaly_window, anomaly_stride, anomaly_threshold],
            outputs=[scorecard_image, scorecard_status],
        )

    with gr.Tab("Change-Point Detection"):
        gr.Markdown(
            """
### Structural Change-Point Detection

Identifies positions where Krylov complexity or entropy changes abruptly,
indicating potential topic shifts, style changes, or structural breaks.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                changepoint_sequence = gr.Textbox(label="Sequence to analyze", lines=5)
                changepoint_window = gr.Slider(
                    minimum=10, maximum=100, value=30, step=1,
                    label="Window size"
                )
                changepoint_detect_btn = gr.Button("Detect change points", variant="primary")
            with gr.Column(scale=2):
                changepoint_output = gr.Textbox(label="Change-point detection results", lines=15)

        changepoint_detect_btn.click(
            fn=detect_change_points,
            inputs=[changepoint_sequence, changepoint_window],
            outputs=[changepoint_output],
        )

    with gr.Tab("Text Generation (Original)"):
        gr.Markdown("## Text Generation")
        with gr.Row():
            with gr.Column(scale=1):
                user_prompt = gr.Textbox(label="Prompt", lines=3)
                paradigm_steps_input = gr.Slider(
                    minimum=0, maximum=50, value=PARADIGM_STEPS, step=1,
                    label="Paradigm steps (n)",
                )
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                corpus_output = gr.Textbox(label="Corpus matches", lines=6)
                generated_output = gr.Textbox(label="Tau model response", lines=8)

        generate_button.click(
            fn=generate_from_prompt,
            inputs=[user_prompt, paradigm_steps_input],
            outputs=[corpus_output, generated_output],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    try:
        reload_globals()
    except FileNotFoundError:
        pass

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
