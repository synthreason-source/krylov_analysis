from __future__ import annotations

import argparse
import json
import math
import random
import re
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

# Contextual generation weights.
CONTEXT_WEIGHT = 1.35
PROMPT_CONTEXT_WEIGHT = 0.45
KRYLOV_CONTEXT_WEIGHT = 0.35
RECENCY_WEIGHT = 0.20
NOVELTY_PENALTY = 0.12

PARADIGM_STEPS = 10
PARADIGM_WEIGHT = 1.2

LEXICAL_WEIGHT = 0.15
VECTOR_WEIGHT = 0.15

IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}

KRYLOV_MAX_ITER = 50
KRYLOV_WINDOW_SIZE = 5
KRYLOV_STRIDE = 1
ANOMALY_THRESHOLD = 2.0

SCORECARD_PATH = "anomaly_scorecard.png"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(float(value), floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
    eps: float = 1e-12,
) -> float:
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
    set_a = set(a) - IGNORED_TOKENS
    set_b = set(b) - IGNORED_TOKENS

    if not set_a or not set_b:
        return 0.0

    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def strip_structural_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in IGNORED_TOKENS]


def _dense_matrix_from_vectors(
    vectors: List[Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for v in vectors for k in v})
    key_index = {k: i for i, k in enumerate(keys)}

    matrix = np.zeros((len(vectors), len(keys)), dtype=np.float64)

    for row, vector in enumerate(vectors):
        for key, weight in vector.items():
            matrix[row, key_index[key]] = weight

    return matrix, keys


def _dominant_eigenvector(
    matrix: np.ndarray,
    iterations: int = 100,
) -> np.ndarray:
    n = matrix.shape[0]

    if n == 0:
        return np.zeros(0)

    vector = np.ones(n, dtype=np.float64) / math.sqrt(n)

    for _ in range(iterations):
        next_vector = matrix @ vector
        norm = np.linalg.norm(next_vector)

        if norm < 1e-12:
            return vector

        next_vector /= norm

        if (
            np.allclose(next_vector, vector, atol=1e-10)
            or np.allclose(next_vector, -vector, atol=1e-10)
        ):
            return next_vector

        vector = next_vector

    return vector


# ---------------------------------------------------------------------------
# Krylov analysis
# ---------------------------------------------------------------------------

@dataclass
class KrylovResult:
    lanczos_a: List[float]
    lanczos_b: List[float]
    complexity_curve: List[float]
    krylov_entropy: float
    anomaly_score: float
    basis_size: int


def _shannon_entropy_bits(v: np.ndarray) -> float:
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
    tolerance: float = 1e-10,
) -> Tuple[List[float], List[float], int]:
    n = operator_matrix.shape[0]

    if n == 0:
        return [], [], 0

    norm = np.linalg.norm(initial_vector)

    if norm < 1e-12:
        return [0.0], [0.0], 1

    v_prev = np.zeros(n, dtype=np.float64)
    v_curr = initial_vector / norm

    a_coeffs: List[float] = []
    b_coeffs: List[float] = []

    b_prev = 0.0

    for iteration in range(max_iter):
        w = operator_matrix @ v_curr

        a_n = float(np.dot(v_curr, w))
        a_coeffs.append(a_n)

        w = w - a_n * v_curr

        if iteration > 0:
            w = w - b_prev * v_prev

        b_n = float(np.linalg.norm(w))
        b_coeffs.append(b_n)

        if b_n < tolerance:
            return a_coeffs, b_coeffs, iteration + 1

        if iteration == max_iter - 1:
            return a_coeffs, b_coeffs, iteration + 1

        v_prev = v_curr.copy()
        v_curr = w / b_n
        b_prev = b_n

    return a_coeffs, b_coeffs, max_iter


def _build_transition_operator(
    model: "NGramModel",
    context_tokens: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    if not model.finalized:
        model.finalize()

    if context_tokens:
        previous = context_tokens[-1]
        previous_previous = (
            context_tokens[-2]
            if len(context_tokens) > 1
            else None
        )
        base_dist = model.backoff_distribution(
            previous,
            previous_previous,
        )
    else:
        base_dist = model.normalize(model.unigram)

    if not base_dist:
        return (
            np.zeros((1, 1), dtype=np.float64),
            ["<unk>"],
            {"<unk>": 0},
        )

    sorted_tokens = sorted(
        base_dist.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    active_tokens = [
        t
        for t, _ in sorted_tokens[:64]
    ]

    if len(active_tokens) < 2:
        token = active_tokens[0] if active_tokens else "<unk>"
        return (
            np.zeros((1, 1), dtype=np.float64),
            [token],
            {token: 0},
        )

    token_to_idx = {
        token: i
        for i, token in enumerate(active_tokens)
    }

    n = len(active_tokens)
    matrix = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    for i, token_i in enumerate(active_tokens):
        next_dist = model.backoff_distribution(
            token_i,
            context_tokens[-1]
            if context_tokens
            else None,
        )

        for j, token_j in enumerate(active_tokens):
            p_ij = max(
                next_dist.get(token_j, 1e-12),
                1e-12,
            )

            p_base = max(
                base_dist.get(token_j, 1e-12),
                1e-12,
            )

            matrix[i, j] = math.log(
                p_ij / p_base
            )

    matrix = (matrix + matrix.T) / 2.0

    return matrix, active_tokens, token_to_idx


def _krylov_dynamic_signal(
    result: KrylovResult,
    candidate_index: int,
) -> float:
    if not result.lanczos_b:
        return 0.0

    usable = [
        abs(x)
        for x in result.lanczos_b
        if math.isfinite(x) and x > 1e-12
    ]

    if not usable:
        return 0.0

    idx = candidate_index % len(usable)
    mean_b = float(np.mean(usable))

    if mean_b < 1e-12:
        return 0.0

    return math.tanh(
        (usable[idx] - mean_b) / mean_b
    )


def _compute_krylov_complexity(
    model: "NGramModel",
    seed_context: List[str],
    max_iter: int = KRYLOV_MAX_ITER,
    time_steps: int = 20,
) -> KrylovResult:
    if not model.finalized:
        model.finalize()

    matrix, tokens, token_to_idx = (
        _build_transition_operator(
            model,
            seed_context,
        )
    )

    dimension = matrix.shape[0]

    if dimension < 2:
        return KrylovResult(
            lanczos_a=[0.0],
            lanczos_b=[0.0],
            complexity_curve=[0.0],
            krylov_entropy=0.0,
            anomaly_score=0.0,
            basis_size=1,
        )

    if (
        seed_context
        and seed_context[-1] in token_to_idx
    ):
        initial = np.zeros(
            dimension,
            dtype=np.float64,
        )
        initial[
            token_to_idx[seed_context[-1]]
        ] = 1.0
    else:
        initial = (
            np.ones(dimension)
            / math.sqrt(dimension)
        )

    a_coeffs, b_coeffs, basis_size = (
        _lanczos_recursion(
            matrix,
            initial,
            max_iter=max_iter,
        )
    )

    basis_dimension = len(a_coeffs)

    if basis_dimension == 0:
        return KrylovResult(
            lanczos_a=[],
            lanczos_b=[],
            complexity_curve=[0.0],
            krylov_entropy=0.0,
            anomaly_score=0.0,
            basis_size=0,
        )

    tridiagonal = np.zeros(
        (basis_dimension, basis_dimension),
        dtype=np.float64,
    )

    for i, value in enumerate(a_coeffs):
        tridiagonal[i, i] = value

    for i in range(
        min(
            basis_dimension - 1,
            len(b_coeffs),
        )
    ):
        value = b_coeffs[i]

        if value > 0:
            tridiagonal[i, i + 1] = value
            tridiagonal[i + 1, i] = value

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            tridiagonal
        )
    except np.linalg.LinAlgError:
        eigenvalues = np.diag(
            tridiagonal
        ).copy()
        eigenvectors = np.eye(
            basis_dimension
        )

    initial_basis = np.zeros(
        basis_dimension,
        dtype=np.float64,
    )
    initial_basis[0] = 1.0

    projected = (
        eigenvectors.T @ initial_basis
    )

    complexity_curve: List[float] = []
    final_probabilities = np.array([1.0])

    for t_index in range(
        max(1, int(time_steps))
    ):
        time = t_index * 0.5

        phase = np.exp(
            -1j * eigenvalues * time
        )

        phi = (
            eigenvectors
            @ (phase * projected)
        )

        probabilities = np.abs(phi) ** 2
        total = float(probabilities.sum())

        if total > 1e-12:
            probabilities /= total

        complexity = float(
            sum(
                index * float(probabilities[index])
                for index in range(
                    len(probabilities)
                )
            )
        )

        complexity_curve.append(
            complexity
        )

        final_probabilities = probabilities

    krylov_entropy = _shannon_entropy_bits(
        final_probabilities
    )

    b_values = [
        float(b)
        for b in b_coeffs
        if b > 1e-6 and math.isfinite(b)
    ]

    if len(b_values) > 2:
        x = np.arange(
            len(b_values),
            dtype=np.float64,
        )
        y = np.asarray(
            b_values,
            dtype=np.float64,
        )

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        residuals = y - (
            slope * x + intercept
        )

        anomaly_score = float(
            np.std(residuals)
            / (np.mean(y) + 1e-6)
        )
    else:
        anomaly_score = 0.0

    return KrylovResult(
        lanczos_a=a_coeffs,
        lanczos_b=b_coeffs,
        complexity_curve=complexity_curve,
        krylov_entropy=krylov_entropy,
        anomaly_score=anomaly_score,
        basis_size=basis_size,
    )


def _sliding_window_krylov(
    model: "NGramModel",
    token_sequence: List[str],
    window_size: int = KRYLOV_WINDOW_SIZE,
    stride: int = KRYLOV_STRIDE,
) -> List[Tuple[int, KrylovResult]]:
    results: List[
        Tuple[int, KrylovResult]
    ] = []

    window_size = max(
        1,
        int(window_size),
    )
    stride = max(
        1,
        int(stride),
    )

    for start in range(
        0,
        len(token_sequence) - window_size + 1,
        stride,
    ):
        window = token_sequence[
            start:start + window_size
        ]

        result = _compute_krylov_complexity(
            model,
            window,
        )

        results.append(
            (start, result)
        )

    return results


def _detect_anomalies(
    krylov_results: List[
        Tuple[int, KrylovResult]
    ],
    threshold: float = ANOMALY_THRESHOLD,
) -> List[Tuple[int, float]]:
    if not krylov_results:
        return []

    anomaly_values = np.asarray(
        [
            result.anomaly_score
            for _, result in krylov_results
        ],
        dtype=np.float64,
    )

    entropy_values = np.asarray(
        [
            result.krylov_entropy
            for _, result in krylov_results
        ],
        dtype=np.float64,
    )

    basis_values = np.asarray(
        [
            result.basis_size
            for _, result in krylov_results
        ],
        dtype=np.float64,
    )

    anomaly_median = float(
        np.median(anomaly_values)
    )
    anomaly_std = float(
        np.std(anomaly_values)
    ) + 1e-6

    entropy_median = float(
        np.median(entropy_values)
    )
    entropy_std = float(
        np.std(entropy_values)
    ) + 1e-6

    basis_median = float(
        np.median(basis_values)
    )

    anomalies: List[
        Tuple[int, float]
    ] = []

    for position, result in krylov_results:
        z_anomaly = (
            result.anomaly_score
            - anomaly_median
        ) / anomaly_std

        z_entropy = (
            abs(
                result.krylov_entropy
                - entropy_median
            )
            / entropy_std
        )

        z_basis = 0.0

        if (
            basis_median > 0
            and result.basis_size
            < 0.5 * basis_median
        ):
            z_basis = 2.0

        combined = max(
            float(z_anomaly),
            float(z_entropy),
            float(z_basis),
        )

        if combined > float(threshold):
            anomalies.append(
                (position, combined)
            )

    return anomalies


# ---------------------------------------------------------------------------
# Corpus search
# ---------------------------------------------------------------------------

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
    def __init__(
        self,
        lexical_weight: float = LEXICAL_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ):
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[
            CorpusReference
        ] = []

    def build_index(
        self,
        corpus_text: str,
    ) -> None:
        sentences = split_sentences(
            corpus_text
        )

        counts = Counter(
            sentence.lower()
            for sentence in sentences
        )

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
                    vector={
                        token: float(count)
                        for token, count
                        in bow.items()
                    },
                    frequency=counts[
                        sentence.lower()
                    ],
                )
            )

    def analyze(
        self,
        prompt: str,
        limit: int = 5,
    ) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)

        prompt_vector = {
            token: float(count)
            for token, count in
            bag_of_words(
                prompt_tokens
            ).items()
        }

        candidates: List[
            Candidate
        ] = []

        for reference in self.references:
            overlap = lexical_overlap(
                prompt_tokens,
                reference.tokens,
            )

            vector_similarity = (
                cosine_similarity(
                    prompt_vector,
                    reference.vector,
                )
            )

            score = (
                self.lexical_weight * overlap
                + self.vector_weight
                * vector_similarity
            )

            candidates.append(
                Candidate(
                    sentence=reference.sentence,
                    symbolic_overlap=overlap,
                    vector_similarity=vector_similarity,
                    frequency=reference.frequency,
                    score=score,
                )
            )

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        candidates = candidates[:limit]

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            candidate.rank = rank

        return candidates


# ---------------------------------------------------------------------------
# Contextual N-gram model
# ---------------------------------------------------------------------------

@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT

    unigram: Counter = field(
        default_factory=Counter
    )

    bigram: Dict[str, Counter] = field(
        default_factory=lambda:
        defaultdict(Counter)
    )

    trigram: Dict[str, Counter] = field(
        default_factory=lambda:
        defaultdict(Counter)
    )

    lexical_vectors: Dict[
        str,
        Dict[str, float],
    ] = field(default_factory=dict)

    vocabulary: List[str] = field(
        default_factory=list
    )

    finalized: bool = False

    def ingest_text(
        self,
        text: str,
    ) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)

            if not words:
                continue

            self._add_sequence(
                [
                    "<bos>",
                    "<bos>",
                    *words,
                    self.eos_token,
                ]
            )

    def _add_sequence(
        self,
        sequence: List[str],
    ) -> None:
        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(
            sequence,
            sequence[1:],
        ):
            self.bigram[left][right] += 1

        for a, b, c in zip(
            sequence,
            sequence[1:],
            sequence[2:],
        ):
            self.trigram[
                f"{a}\t{b}"
            ][c] += 1

        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(
            token
            for token, count
            in self.unigram.items()
            if count >= self.min_count
        )

        if (
            self.unk_token
            not in self.vocabulary
        ):
            self.vocabulary.append(
                self.unk_token
            )

        token_contexts: Dict[
            str,
            Counter,
        ] = defaultdict(Counter)

        for context, counts in (
            self.bigram.items()
        ):
            for token, count in (
                counts.items()
            ):
                token_contexts[
                    token
                ][context] += count

        self.lexical_vectors = {}

        for token in self.vocabulary:
            counts = token_contexts.get(
                token,
                Counter(),
            )

            total = sum(counts.values())

            if total <= 0:
                self.lexical_vectors[
                    token
                ] = {}
            else:
                self.lexical_vectors[
                    token
                ] = {
                    context: count / total
                    for context, count
                    in counts.items()
                }

        self.finalized = True

    def prompt_kernel_vector(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        tokens = [
            token
            for token in tokenize(prompt)
            if token not in IGNORED_TOKENS
            and self.lexical_vectors.get(token)
        ]

        if not tokens:
            return {}

        vectors = [
            self.lexical_vectors[token]
            for token in tokens
        ]

        matrix, keys = (
            _dense_matrix_from_vectors(
                vectors
            )
        )

        if matrix.size == 0:
            return {}

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        norms[
            norms < 1e-12
        ] = 1.0

        normalized = matrix / norms
        kernel = normalized @ normalized.T

        weights = np.abs(
            _dominant_eigenvector(
                kernel
            )
        )

        total = float(weights.sum())

        if total < 1e-12:
            return {}

        weights /= total

        paradigm = weights @ matrix

        return {
            key: float(value)
            for key, value in zip(
                keys,
                paradigm,
            )
            if abs(value) > 1e-9
        }

    def context_vector(
        self,
        context_tokens: List[str],
        max_tokens: int = 8,
    ) -> Dict[str, float]:
        """
        Build a recency-weighted second-order context vector.

        A candidate token has a learned lexical vector describing the contexts
        in which it appears. The current sequence is transformed into a
        comparable vector, so generation can favor conceptual continuity
        rather than simply the most grammatical continuation.
        """
        tokens = [
            token
            for token in context_tokens[
                -max_tokens:
            ]
            if token not in IGNORED_TOKENS
        ]

        if not tokens:
            return {}

        accumulated: Dict[
            str,
            float,
        ] = defaultdict(float)

        count = len(tokens)

        for position, token in enumerate(tokens):
            vector = self.lexical_vectors.get(
                token,
                {},
            )

            if not vector:
                continue

            age = count - position

            weight = 1.0 / (
                0.75 + 0.35 * age
            )

            for key, value in vector.items():
                accumulated[key] += (
                    weight * value
                )

        norm = math.sqrt(
            sum(
                value * value
                for value
                in accumulated.values()
            )
        )

        if norm < 1e-12:
            return {}

        return {
            key: value / norm
            for key, value
            in accumulated.items()
        }

    def token_context_vector(
        self,
        token: str,
    ) -> Dict[str, float]:
        vector = self.lexical_vectors.get(
            token,
            {},
        )

        if not vector:
            return {}

        norm = math.sqrt(
            sum(
                value * value
                for value
                in vector.values()
            )
        )

        if norm < 1e-12:
            return {}

        return {
            key: value / norm
            for key, value
            in vector.items()
        }

    def backoff_distribution(
        self,
        previous: str,
        previous_previous: Optional[str],
    ) -> Dict[str, float]:
        if previous_previous is not None:
            counts = self.trigram.get(
                f"{previous_previous}\t{previous}"
            )

            if counts:
                return self.normalize(
                    counts
                )

        counts = self.bigram.get(
            previous
        )

        if counts:
            return self.normalize(
                counts
            )

        return self.normalize(
            self.unigram
        )

    @staticmethod
    def normalize(
        counts: Counter,
    ) -> Dict[str, float]:
        total = sum(counts.values())

        if total <= 0:
            return {}

        return {
            token: count / total
            for token, count
            in counts.items()
        }

    def resolve_context(
        self,
        prompt: str,
    ) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)

        if not tokens:
            return "<bos>", None

        previous = tokens[-1]

        previous_previous = (
            tokens[-2]
            if len(tokens) >= 2
            else None
        )

        return (
            previous,
            previous_previous,
        )

    def _contextual_candidate_score(
        self,
        candidate: str,
        generated_tokens: List[str],
        context_vector: Optional[
            Dict[str, float]
        ] = None,
        prompt_vector: Optional[
            Dict[str, float]
        ] = None,
        krylov_result: Optional[
            KrylovResult
        ] = None,
        candidate_rank: int = 0,
    ) -> float:
        recent = [
            token
            for token in generated_tokens[-8:]
            if token not in IGNORED_TOKENS
        ]

        if context_vector is None:
            context_vector = self.context_vector(
                recent
            )

        candidate_vector = (
            self.token_context_vector(
                candidate
            )
        )

        local_similarity = cosine_similarity(
            context_vector,
            candidate_vector,
        )

        prompt_similarity = 0.0

        if prompt_vector:
            prompt_similarity = cosine_similarity(
                prompt_vector,
                candidate_vector,
            )

        krylov_signal = 0.0

        if krylov_result is not None:
            candidate_index = (
                sum(
                    ord(char)
                    for char in candidate
                )
                + candidate_rank
            )

            krylov_signal = (
                _krylov_dynamic_signal(
                    krylov_result,
                    candidate_index,
                )
            )

        repetition_count = recent.count(
            candidate
        )

        repetition_penalty = (
            NOVELTY_PENALTY
            * max(
                0,
                repetition_count - 1,
            )
        )

        recency_signal = 0.0

        if recent:
            previous = recent[-1]

            pair_distribution = (
                self.backoff_distribution(
                    previous,
                    recent[-2]
                    if len(recent) >= 2
                    else None,
                )
            )

            pair_probability = (
                pair_distribution.get(
                    candidate,
                    0.0,
                )
            )

            recency_signal = (
                math.log1p(
                    pair_probability * 100.0
                )
                / 5.0
            )

        return (
            CONTEXT_WEIGHT
            * local_similarity
            + PROMPT_CONTEXT_WEIGHT
            * prompt_similarity
            + KRYLOV_CONTEXT_WEIGHT
            * krylov_signal
            + RECENCY_WEIGHT
            * recency_signal
            - repetition_penalty
        )

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        instruction_vector: Optional[
            Dict[str, float]
        ] = None,
        instruction_weight: float = 0.0,
        krylov_result: Optional[
            KrylovResult
        ] = None,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        tokens = tokenize(prompt)

        previous, previous_previous = (
            self.resolve_context(prompt)
        )

        base = self.backoff_distribution(
            previous,
            previous_previous,
        )

        if not base:
            return {}

        candidates = sorted(
            base,
            key=base.get,
            reverse=True,
        )[:max(
            1,
            int(candidate_limit),
        )]

        # Dynamic context vector is calculated once for this generation step.
        context_vector = self.context_vector(
            tokens[-8:]
        )

        scores: Dict[str, float] = {}

        for rank, token in enumerate(
            candidates
        ):
            score = safe_log(
                base[token]
            )

            score += (
                self._contextual_candidate_score(
                    token,
                    tokens,
                    context_vector=context_vector,
                    prompt_vector=instruction_vector,
                    krylov_result=krylov_result,
                    candidate_rank=rank,
                )
            )

            if (
                instruction_vector
                and instruction_weight
            ):
                similarity = cosine_similarity(
                    instruction_vector,
                    self.lexical_vectors.get(
                        token,
                        {},
                    ),
                )

                score += (
                    instruction_weight
                    * similarity
                )

            scores[token] = score

        return scores

    def probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        instruction_vector: Optional[
            Dict[str, float]
        ] = None,
        instruction_weight: float = 0.0,
        krylov_result: Optional[
            KrylovResult
        ] = None,
    ) -> Dict[str, float]:
        scores = self.score_next_token(
            prompt,
            candidate_limit,
            instruction_vector,
            instruction_weight,
            krylov_result,
        )

        if not scores:
            return {}

        temperature = max(
            float(temperature),
            1e-5,
        )

        scaled = {
            token: score / temperature
            for token, score
            in scores.items()
        }

        maximum = max(
            scaled.values()
        )

        exponentials = {
            token: math.exp(
                score - maximum
            )
            for token, score
            in scaled.items()
        }

        total = sum(
            exponentials.values()
        )

        if total <= 0:
            return {}

        return {
            token: value / total
            for token, value
            in exponentials.items()
        }

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[
            Dict[str, float]
        ] = None,
        instruction_weight: float = 0.0,
        krylov_result: Optional[
            KrylovResult
        ] = None,
    ) -> str:
        probabilities = self.probabilities(
            prompt,
            temperature,
            max(
                int(top_k),
                1,
            ),
            instruction_vector,
            instruction_weight,
            krylov_result,
        )

        if not probabilities:
            return self.eos_token

        items = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max(
            1,
            int(top_k),
        )]

        candidate_tokens, weights = zip(
            *items
        )

        return random.choices(
            candidate_tokens,
            weights=weights,
            k=1,
        )[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[
            Dict[str, float]
        ] = None,
        instruction_weight: float = 0.0,
        paradigm_steps: int = 0,
        paradigm_weight: float = PARADIGM_WEIGHT,
    ) -> str:
        """
        Context-first generation.

        The contextual state is refreshed throughout generation. The original
        prompt is not treated as the only source of meaning: the sequence that
        has actually emerged becomes the active context.
        """
        generated = tokenize(prompt)

        if not generated:
            generated = ["<bos>"]

        paradigm_vector = (
            self.prompt_kernel_vector(prompt)
            if int(paradigm_steps) > 0
            else None
        )

        krylov_result: Optional[
            KrylovResult
        ] = None

        refresh_every = 3

        for step in range(
            max(
                0,
                int(max_new_tokens),
            )
        ):
            if (
                step % refresh_every == 0
            ):
                recent_context = [
                    token
                    for token in generated[-8:]
                    if token not in IGNORED_TOKENS
                ]

                if recent_context:
                    krylov_result = (
                        _compute_krylov_complexity(
                            self,
                            recent_context,
                            max_iter=min(
                                KRYLOV_MAX_ITER,
                                24,
                            ),
                            time_steps=10,
                        )
                    )

            if (
                paradigm_vector
                and step < int(paradigm_steps)
            ):
                active_vector = (
                    paradigm_vector
                )
                active_weight = (
                    paradigm_weight
                )
            else:
                active_vector = (
                    instruction_vector
                )
                active_weight = (
                    instruction_weight
                )

            token = self.sample_next(
                " ".join(generated),
                temperature=temperature,
                top_k=top_k,
                instruction_vector=active_vector,
                instruction_weight=active_weight,
                krylov_result=krylov_result,
            )

            if token == self.eos_token:
                break

            generated.append(token)

        return self.detokenize(
            strip_structural_tokens(
                generated
            )
        )

    @staticmethod
    def detokenize(
        tokens: List[str],
    ) -> str:
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "unigram": dict(
                self.unigram
            ),
            "bigram": {
                key: dict(value)
                for key, value
                in self.bigram.items()
            },
            "trigram": {
                key: dict(value)
                for key, value
                in self.trigram.items()
            },
            "lexical_vectors": (
                self.lexical_vectors
            ),
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "NGramModel":
        model = cls(
            eos_token=data.get(
                "eos_token",
                "<eos>",
            ),
            unk_token=data.get(
                "unk_token",
                "<unk>",
            ),
            min_count=data.get(
                "min_count",
                MIN_COUNT,
            ),
        )

        model.unigram = Counter(
            data.get(
                "unigram",
                {},
            )
        )

        model.bigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value
                in data.get(
                    "bigram",
                    {},
                ).items()
            },
        )

        model.trigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value
                in data.get(
                    "trigram",
                    {},
                ).items()
            },
        )

        model.lexical_vectors = (
            data.get(
                "lexical_vectors",
                {},
            )
        )

        model.vocabulary = data.get(
            "vocabulary",
            [],
        )

        model.finalized = data.get(
            "finalized",
            False,
        )

        return model

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "NGramModel":
        return cls.from_dict(
            json.loads(
                Path(path).read_text(
                    encoding="utf-8"
                )
            )
        )


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

TEXT_MODEL: Optional[NGramModel] = None
CORPUS_SEARCH: Optional[CorpusSearch] = None
CORPUS_TEXT_CACHE: Optional[str] = None


def load_text_model() -> NGramModel:
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. "
            "Upload a corpus and click 'Train model' first."
        )

    model = NGramModel.load_json(
        model_path
    )

    if not model.finalized:
        model.finalize()

    return model


def load_corpus_search(
    corpus_text: str,
) -> CorpusSearch:
    search = CorpusSearch()
    search.build_index(corpus_text)
    return search


def reload_globals() -> None:
    global TEXT_MODEL
    global CORPUS_SEARCH
    global CORPUS_TEXT_CACHE

    TEXT_MODEL = load_text_model()

    if CORPUS_TEXT_CACHE:
        corpus_text = CORPUS_TEXT_CACHE
    else:
        default_path = Path(
            DEFAULT_CORPUS_FILE
        )

        if default_path.exists():
            corpus_text = (
                default_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
        else:
            corpus_text = ""

        CORPUS_TEXT_CACHE = corpus_text

    CORPUS_SEARCH = load_corpus_search(
        corpus_text
    )


def format_matches(
    prompt: str,
    limit: int = 5,
) -> str:
    if (
        CORPUS_SEARCH is None
        or not CORPUS_SEARCH.references
    ):
        return "No corpus file loaded."

    candidates = CORPUS_SEARCH.analyze(
        prompt,
        limit=limit,
    )

    if not candidates:
        return "No corpus matches found."

    return "\n".join(
        (
            f"{candidate.rank}. "
            f"{candidate.sentence} "
            f"(score={candidate.score:.3f}, "
            f"overlap="
            f"{candidate.symbolic_overlap:.3f}, "
            f"vector="
            f"{candidate.vector_similarity:.3f})"
        )
        for candidate in candidates
    )


def train_model_from_file(
    corpus_file,
):
    global CORPUS_TEXT_CACHE

    if corpus_file is None:
        return (
            "No corpus file uploaded.",
            "",
            "",
        )

    path = (
        Path(corpus_file.name)
        if hasattr(
            corpus_file,
            "name",
        )
        else Path(corpus_file)
    )

    if not path.exists():
        return (
            f"Corpus file not found: {path}",
            "",
            "",
        )

    corpus_text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    CORPUS_TEXT_CACHE = corpus_text

    model = NGramModel()
    model.ingest_text(corpus_text)
    model.finalize()
    model.save_json(
        Path(MODEL_PATH)
    )

    reload_globals()

    sample = "\n".join(
        corpus_text.splitlines()[:5]
    )

    summary = (
        f"Vocabulary: "
        f"{len(model.vocabulary)}\n"
        f"Unigrams: "
        f"{len(model.unigram)}\n"
        f"Bigram contexts: "
        f"{len(model.bigram)}\n"
        f"Trigram contexts: "
        f"{len(model.trigram)}\n"
        f"Contextual generation: enabled\n"
        f"Krylov contextual signal: enabled"
    )

    return (
        f"Model trained and saved to "
        f"{MODEL_PATH}.",
        sample,
        summary,
    )


def generate_from_prompt(
    user_prompt: str,
    paradigm_steps: int,
):
    if (
        TEXT_MODEL is None
        or CORPUS_SEARCH is None
    ):
        return (
            "Model not loaded. Upload a corpus "
            "and click 'Train model' first.",
            "",
        )

    prompt = (
        user_prompt.strip()
        if user_prompt
        and user_prompt.strip()
        else "<bos>"
    )

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        paradigm_steps=int(
            paradigm_steps
        ),
        paradigm_weight=PARADIGM_WEIGHT,
    )

    matches = (
        format_matches(
            prompt,
            limit=5,
        )
        if prompt != "<bos>"
        else "No prompt provided for corpus search."
    )

    return matches, generated


# ---------------------------------------------------------------------------
# Krylov UI functions
# ---------------------------------------------------------------------------

def analyze_krylov_single(
    prompt: str,
    max_iter: int,
) -> str:
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(prompt)

    if not tokens:
        return "Empty prompt."

    result = _compute_krylov_complexity(
        TEXT_MODEL,
        tokens,
        max_iter=int(max_iter),
    )

    lines = [
        f"Krylov Analysis for: "
        f"'{prompt[:80]}'",
        "",
        f"Basis size: "
        f"{result.basis_size}",
        (
            "Lanczos a: "
            f"{[round(x, 4) for x in result.lanczos_a[:10]]}"
            f"{'...' if len(result.lanczos_a) > 10 else ''}"
        ),
        (
            "Lanczos b: "
            f"{[round(x, 4) for x in result.lanczos_b[:10]]}"
            f"{'...' if len(result.lanczos_b) > 10 else ''}"
        ),
        "",
        f"Krylov entropy: "
        f"{result.krylov_entropy:.4f} bits",
        f"Anomaly score: "
        f"{result.anomaly_score:.4f}",
        "",
        "Complexity curve K(t):",
    ]

    for index, value in enumerate(
        result.complexity_curve
    ):
        lines.append(
            f"  t={index}: K={value:.4f}"
        )

    lines.extend(
        [
            "",
            "The anomaly signal uses learned "
            "transition dynamics and Krylov "
            "spreading rather than grammatical "
            "correctness alone.",
        ]
    )

    return "\n".join(lines)


def analyze_krylov_sequence(
    sequence_text: str,
    window_size: int,
    stride: int,
    threshold: float,
) -> str:
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(sequence_text)

    window_size = int(window_size)
    stride = int(stride)

    if len(tokens) < window_size:
        return (
            f"Sequence too short "
            f"({len(tokens)} tokens). "
            f"Need at least {window_size}."
        )

    results = _sliding_window_krylov(
        TEXT_MODEL,
        tokens,
        window_size=window_size,
        stride=stride,
    )

    if not results:
        return "No windows analyzed."

    anomalies = _detect_anomalies(
        results,
        threshold=float(threshold),
    )

    anomaly_positions = {
        position
        for position, _ in anomalies
    }

    lines = [
        "Krylov Sequence Analysis",
        f"Sequence length: "
        f"{len(tokens)} tokens",
        f"Window size: "
        f"{window_size}, stride: {stride}",
        "",
        "Window-by-window results:",
    ]

    for start, result in results:
        window_tokens = tokens[
            start:start + window_size
        ]

        preview = " ".join(
            window_tokens[:10]
        )

        if len(window_tokens) > 10:
            preview += "..."

        flag = (
            " [CONTEXTUAL ANOMALY]"
            if start in anomaly_positions
            else ""
        )

        final_k = (
            result.complexity_curve[-1]
            if result.complexity_curve
            else 0.0
        )

        lines.append(
            f"\n  Window {start}-"
            f"{start + window_size}: "
            f"{preview}"
            f"\n    K={final_k:.3f}, "
            f"S={result.krylov_entropy:.3f}, "
            f"anomaly="
            f"{result.anomaly_score:.3f}"
            f"{flag}"
        )

    if anomalies:
        lines.append("")
        lines.append(
            f"Detected {len(anomalies)} "
            "contextual anomalies:"
        )

        for position, score in anomalies:
            preview = tokens[
                position:
                position + min(
                    10,
                    len(tokens) - position,
                )
            ]

            lines.append(
                f"  Position {position}: "
                f"{' '.join(preview)} "
                f"(context score={score:.3f})"
            )
    else:
        lines.extend(
            [
                "",
                "No contextual anomalies detected.",
            ]
        )

    return "\n".join(lines)


def detect_change_points(
    sequence_text: str,
    window_size: int,
) -> str:
    if TEXT_MODEL is None:
        return "Model not loaded."

    tokens = tokenize(sequence_text)
    window_size = int(window_size)

    if len(tokens) < 2 * window_size:
        return (
            "Sequence too short for "
            "change-point detection."
        )

    stride = max(
        1,
        window_size // 4,
    )

    results = _sliding_window_krylov(
        TEXT_MODEL,
        tokens,
        window_size=window_size,
        stride=stride,
    )

    if len(results) < 3:
        return (
            "Not enough windows for "
            "change-point detection."
        )

    change_points = []
    previous = None

    for start, result in results:
        if previous is not None:
            delta_k = abs(
                result.complexity_curve[-1]
                - previous.complexity_curve[-1]
            )

            delta_entropy = abs(
                result.krylov_entropy
                - previous.krylov_entropy
            )

            delta_anomaly = abs(
                result.anomaly_score
                - previous.anomaly_score
            )

            if (
                delta_k > 0.5
                or delta_entropy > 0.3
                or delta_anomaly > 1.0
            ):
                change_points.append(
                    (
                        start,
                        delta_k,
                        delta_entropy,
                        delta_anomaly,
                    )
                )

        previous = result

    lines = [
        "Change-Point Detection Results",
        (
            f"Sequence: {len(tokens)} tokens, "
            f"window size: {window_size}"
        ),
        "",
    ]

    if change_points:
        lines.append(
            f"Found {len(change_points)} "
            "potential contextual change points:"
        )

        for (
            position,
            delta_k,
            delta_entropy,
            delta_anomaly,
        ) in change_points:
            context = tokens[
                max(0, position - 2):
                position + min(
                    8,
                    len(tokens) - position,
                )
            ]

            lines.append(
                f"  Position {position}: "
                f"...{' '.join(context)}..."
                f"\n    ΔK={delta_k:.3f}, "
                f"ΔS={delta_entropy:.3f}, "
                f"Δanomaly={delta_anomaly:.3f}"
            )
    else:
        lines.extend(
            [
                "No significant contextual "
                "change points detected.",
                "",
                "The sequence appears structurally "
                "homogeneous by Krylov metrics.",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def _bracket_anomalous_windows(
    tokens: List[str],
    window_size: int,
    anomaly_positions: Iterable[int],
) -> str:
    covered = [
        False
        for _ in tokens
    ]

    for position in anomaly_positions:
        for index in range(
            position,
            min(
                position + window_size,
                len(tokens),
            ),
        ):
            covered[index] = True

    pieces: List[str] = []
    index = 0

    while index < len(tokens):
        if covered[index]:
            end = index
            run: List[str] = []

            while (
                end < len(tokens)
                and covered[end]
            ):
                run.append(tokens[end])
                end += 1

            pieces.append(
                "[" + " ".join(run) + "]"
            )
            index = end
        else:
            pieces.append(tokens[index])
            index += 1

    return " ".join(pieces)


def _draw_scorecard_png(
    annotated_text: str,
    score_rows: List[Dict[str, Any]],
    threshold: float,
    anomaly_count: int,
    window_count: int,
    output_path: str,
) -> None:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, (
        text_axis,
        table_axis,
    ) = plt.subplots(
        2,
        1,
        figsize=(11, 8.5),
        gridspec_kw={
            "height_ratios": [3, 2]
        },
    )

    fig.suptitle(
        "Krylov Contextual Anomaly Scorecard",
        fontsize=17,
        fontweight="bold",
    )

    text_axis.axis("off")

    text_axis.set_title(
        (
            f"Scanned text — "
            f"{anomaly_count} of "
            f"{window_count} windows flagged "
            "([bracketed] = contextual anomaly)"
        ),
        fontsize=11,
        loc="left",
    )

    wrapped = textwrap.fill(
        annotated_text,
        width=78,
    )

    text_axis.text(
        0.0,
        1.0,
        wrapped,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
        transform=text_axis.transAxes,
    )

    table_axis.axis("off")

    table_axis.set_title(
        (
            f"Per-window scores "
            f"(flag threshold = {threshold})"
        ),
        fontsize=11,
        loc="left",
    )

    columns = [
        "start",
        "K(t)",
        "entropy (bits)",
        "anomaly score",
        "flag",
    ]

    cells = []

    for row in score_rows:
        cells.append(
            [
                str(row["start"]),
                f"{row['K']:.3f}",
                f"{row['entropy']:.3f}",
                f"{row['anomaly_score']:.3f}",
                (
                    "contextual anomaly"
                    if row["flagged"]
                    else ""
                ),
            ]
        )

    table = table_axis.table(
        cellText=cells,
        colLabels=columns,
        loc="upper center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_anomaly_scorecard(
    sequence_text: str,
    window_size: int,
    stride: int,
    threshold: float,
) -> Tuple[Optional[str], str]:
    if TEXT_MODEL is None:
        return None, "Model not loaded."

    tokens = tokenize(sequence_text)

    window_size = int(window_size)
    stride = int(stride)

    if len(tokens) < window_size:
        return (
            None,
            f"Sequence too short "
            f"({len(tokens)} tokens). "
            f"Need at least {window_size}.",
        )

    results = _sliding_window_krylov(
        TEXT_MODEL,
        tokens,
        window_size=window_size,
        stride=stride,
    )

    if not results:
        return None, "No windows analyzed."

    anomalies = _detect_anomalies(
        results,
        threshold=float(threshold),
    )

    anomaly_lookup = dict(anomalies)

    annotated = _bracket_anomalous_windows(
        tokens,
        window_size,
        anomaly_lookup.keys(),
    )

    rows = [
        {
            "start": start,
            "K": (
                result.complexity_curve[-1]
                if result.complexity_curve
                else 0.0
            ),
            "entropy": result.krylov_entropy,
            "anomaly_score": result.anomaly_score,
            "flagged": (
                start in anomaly_lookup
            ),
        }
        for start, result in results
    ]

    output_path = str(
        Path(
            SCORECARD_PATH
        ).resolve()
    )

    _draw_scorecard_png(
        annotated,
        rows,
        float(threshold),
        anomaly_count=len(
            anomaly_lookup
        ),
        window_count=len(results),
        output_path=output_path,
    )

    return (
        output_path,
        (
            f"{len(anomaly_lookup)} of "
            f"{len(results)} windows flagged "
            "as contextual anomalies. "
            "Scorecard saved."
        ),
    )


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Krylov Contextual Detector"
) as demo:

    gr.Markdown(
        """
# Krylov Complexity Detector

**Context-first n-gram generation + Krylov sequence analysis**

The generator follows the evolving contextual trajectory of the text rather
than treating grammatical probability as the sole criterion.
"""
    )

    with gr.Tab(
        "Corpus & Training"
    ):
        with gr.Row():
            with gr.Column(scale=1):
                corpus_file_input = gr.File(
                    label="Corpus file",
                    file_types=["file"],
                )

                train_button = gr.Button(
                    "Train model",
                    variant="primary",
                )

            with gr.Column(scale=2):
                train_status = gr.Textbox(
                    label="Training status",
                    lines=2,
                )

                corpus_sample = gr.Textbox(
                    label="Corpus sample",
                    lines=5,
                )

                model_summary = gr.Textbox(
                    label="Model summary",
                    lines=7,
                )

        train_button.click(
            fn=train_model_from_file,
            inputs=[corpus_file_input],
            outputs=[
                train_status,
                corpus_sample,
                model_summary,
            ],
        )

    with gr.Tab(
        "Single Prompt Analysis"
    ):
        with gr.Row():
            with gr.Column(scale=1):
                krylov_prompt = gr.Textbox(
                    label="Prompt to analyze",
                    lines=3,
                )

                krylov_max_iter = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=30,
                    step=1,
                    label="Max Lanczos iterations",
                )

                krylov_analyze_button = (
                    gr.Button(
                        "Analyze",
                        variant="primary",
                    )
                )

            with gr.Column(scale=2):
                krylov_output = gr.Textbox(
                    label="Krylov analysis results",
                    lines=18,
                )

        krylov_analyze_button.click(
            fn=analyze_krylov_single,
            inputs=[
                krylov_prompt,
                krylov_max_iter,
            ],
            outputs=[krylov_output],
        )

    with gr.Tab(
        "Sequence Anomaly Detection"
    ):
        gr.Markdown(
            """
The anomaly detector is based on learned transition dynamics and Krylov
spreading. It does not label text anomalous simply because the grammar is
unusual.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                anomaly_sequence = gr.Textbox(
                    label="Sequence to scan",
                    lines=7,
                )

                anomaly_window = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=5,
                    step=1,
                    label="Window size",
                )

                anomaly_stride = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="Stride",
                )

                anomaly_threshold = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    label="Contextual anomaly threshold",
                )

                anomaly_detect_button = (
                    gr.Button(
                        "Detect contextual anomalies",
                        variant="primary",
                    )
                )

                scorecard_button = gr.Button(
                    "Generate scorecard PNG"
                )

            with gr.Column(scale=2):
                anomaly_output = gr.Textbox(
                    label="Anomaly detection results",
                    lines=18,
                )

                scorecard_image = gr.Image(
                    label="Anomaly scorecard",
                    type="filepath",
                )

                scorecard_status = gr.Textbox(
                    label="Scorecard status",
                    lines=2,
                )

        anomaly_detect_button.click(
            fn=analyze_krylov_sequence,
            inputs=[
                anomaly_sequence,
                anomaly_window,
                anomaly_stride,
                anomaly_threshold,
            ],
            outputs=[anomaly_output],
        )

        scorecard_button.click(
            fn=generate_anomaly_scorecard,
            inputs=[
                anomaly_sequence,
                anomaly_window,
                anomaly_stride,
                anomaly_threshold,
            ],
            outputs=[
                scorecard_image,
                scorecard_status,
            ],
        )

    with gr.Tab(
        "Change-Point Detection"
    ):
        with gr.Row():
            with gr.Column(scale=1):
                changepoint_sequence = gr.Textbox(
                    label="Sequence to analyze",
                    lines=7,
                )

                changepoint_window = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=1,
                    label="Window size",
                )

                changepoint_button = gr.Button(
                    "Detect change points",
                    variant="primary",
                )

            with gr.Column(scale=2):
                changepoint_output = gr.Textbox(
                    label="Change-point detection results",
                    lines=18,
                )

        changepoint_button.click(
            fn=detect_change_points,
            inputs=[
                changepoint_sequence,
                changepoint_window,
            ],
            outputs=[changepoint_output],
        )

    with gr.Tab(
        "Text Generation"
    ):
        gr.Markdown(
            """
## Contextual Text Generation

Every generation step uses the evolving sequence as context.

The scoring combines:
- trigram/bigram/unigram backoff;
- a recency-weighted contextual vector;
- candidate-to-context vector similarity;
- prompt/kernel similarity;
- a lightweight Krylov transition signal;
- soft repetition control.

This is intended to make generation contextual rather than merely
grammatical.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                user_prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                )

                paradigm_steps_input = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=PARADIGM_STEPS,
                    step=1,
                    label="Initial paradigm/context steps",
                )

                generate_button = gr.Button(
                    "Generate",
                    variant="primary",
                )

            with gr.Column(scale=2):
                corpus_output = gr.Textbox(
                    label="Corpus contextual matches",
                    lines=7,
                )

                generated_output = gr.Textbox(
                    label="Contextual Krylov response",
                    lines=14,
                )

        generate_button.click(
            fn=generate_from_prompt,
            inputs=[
                user_prompt,
                paradigm_steps_input,
            ],
            outputs=[
                corpus_output,
                generated_output,
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--share",
        action="store_true",
    )

    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
    )

    args = parser.parse_args()

    try:
        reload_globals()
    except FileNotFoundError:
        print(
            "No model.json found yet. "
            "Upload a corpus and train the model."
        )
    except Exception as exc:
        print(
            "Warning: could not load existing model: "
            f"{exc}"
        )

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
