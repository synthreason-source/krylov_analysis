# Krylov Complexity Detector

A small n-gram language model with a Gradio UI, extended with a Krylov-subspace
anomaly detector for spotting structural/topical breaks in text.

Pure Python + NumPy. No torch, no GPU required.

## What's in here

The app has two layers built on the same trained n-gram model:

1. **Text generation kernel** — a trigram-backoff language model with a
   linear-algebra "paradigm" mechanism that keeps generated text anchored to
   the topic of the prompt.
2. **Krylov anomaly detector** — treats the trained model's token-transition
   statistics as an operator, runs the Lanczos algorithm on it, and uses the
   resulting spectral behavior to flag windows of text that look
   structurally different from their surroundings.

## Setup

```bash
pip install gradio numpy matplotlib
python krylov_app.py
```

This launches a local Gradio server (default `http://127.0.0.1:7860`).
Use `--share` to get a temporary public link, and `--server-port` /
`--server-name` to change the bind address.

## 1. Corpus & training

Upload any text file as the corpus and click **Train model**. This:

- Splits the corpus into sentences (on `.`) and tokenizes them (lowercase,
  whitespace split).
- Builds unigram, bigram, and trigram counts.
- Builds a "lexical vector" for each token: the normalized distribution of
  bigram contexts it tends to follow.
- Saves everything to `model.json`, which is what every other tab reads from.

You need to train a model before using any of the other tabs.

## 2. Text generation

Trigram → bigram → unigram backoff sampling, with two extras:

- **Paradigm steps** — before generating, the prompt's own tokens are
  turned into a Gram (kernel) matrix of pairwise cosine similarities, and
  its dominant eigenvector (via power iteration) is projected back into a
  single "paradigm vector" — the semantic center of the prompt. The first
  *n* generated tokens are biased toward it, so the reply opens grounded in
  what was actually said instead of drifting from token one.
- **Sentences** — generation runs sentence-by-sentence. Each sentence
  can be told to close on its own once its running content circles back
  into agreement with the paradigm vector (a symmetric Gram-matrix
  similarity check), rather than always running to a fixed token cap. The
  next sentence starts fresh but stays anchored to the same paradigm
  vector, so a multi-sentence paragraph stays on-topic while each sentence
  is free to phrase it differently.

## 3. Krylov anomaly detection

The core idea: build a small operator matrix `L` from the model's own
next-token statistics around a window of text, run the **Lanczos
algorithm** on it to get coefficients `{a_n, b_n}`, and use those
coefficients to characterize how "chaotic" vs. "boring" that window's local
dynamics are. Three tabs use this:

- **Single Prompt Analysis** — runs Lanczos on one prompt and shows the raw
  coefficients, a complexity curve `K(t)`, Krylov entropy, and an anomaly
  score.
- **Sequence Anomaly Detection** — slides a window across a longer
  sequence, computes the same metrics for every window, and flags windows
  whose combined z-score (across anomaly score, entropy, and Krylov basis
  size) exceeds a threshold.
  - **Generate scorecard PNG** renders the *entire* scanned text as one
    image, with every flagged window's words wrapped in `[ ]` directly in
    place among the rest of the text — no separate table, just the
    annotated passage plus a one-line summary of how many windows were
    flagged.
- **Change-Point Detection** — looks for sudden jumps in `K(t)`, entropy,
  or anomaly score between consecutive windows, as a way to locate topic
  shifts or style breaks in a longer document.

### Why you'll sometimes see all-zero scores

If a window's local context is highly predictable (e.g. "the cat" is
almost always followed by the same word or two in a small corpus), the
Lanczos recursion can terminate after a single iteration — there's nowhere
for the vector to "spread" to. That collapses `K(t)`, entropy, and the
anomaly score to `0.0` for that window. It's not a bug; it means the
model found that stretch of text locally deterministic. Windows with real
branching/ambiguity in their local context are the ones that produce
non-trivial (and potentially anomalous) scores — this is also why the
detector tends to work better on larger, more varied corpora than tiny toy
ones.

## Files

- `krylov_app.py` — the whole app (model, detector, Gradio UI).
- `model.json` — created after training; the persisted n-gram model.
- `anomaly_scorecard.png` — created when you click **Generate scorecard
  PNG**; overwritten on each click.

## Parameters worth knowing about

| Constant | Meaning |
|---|---|
| `PARADIGM_STEPS` / `PARADIGM_WEIGHT` | How many opening tokens of each sentence get biased toward the prompt's paradigm vector, and how strongly. |
| `TOPIC_CLOSURE_TAU` / `MIN_STEPS_BEFORE_CLOSURE` | Similarity threshold and minimum sentence length before a sentence is allowed to close early. |
| `KRYLOV_MAX_ITER` | Cap on Lanczos iterations per window (bounds the max Krylov subspace dimension considered). |
| `KRYLOV_WINDOW_SIZE` / `KRYLOV_STRIDE` | Default sliding-window size and step for sequence scanning. |
| `ANOMALY_THRESHOLD` | Default z-score threshold for flagging a window as anomalous. |

All of these are also exposed as sliders/inputs in the relevant UI tab.
