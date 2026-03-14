# Architecture

This document describes the high-level architecture of the NLP Pipeline and how each component relates to the others.

## Overview

The project implements a full NLP pipeline progressing from classical text processing to modern neural embeddings:

```
Raw Text → Preprocessing → Language Models → Tokenization → Embeddings → Evaluation
```

## Component Architecture

```
src/nlp_pipeline/
├── preprocessing.py      # Text acquisition and cleaning
├── language_models.py     # Statistical language models
├── bpe.py                 # Subword tokenization
└── embeddings.py          # Dense and sparse representations
```

## Module Details

### 1. Preprocessing (`preprocessing.py`)

Handles the first stage of any NLP pipeline — getting clean text from raw sources.

**Capabilities:**
- **Web scraping** — `fetch_clean_text()` downloads and cleans HTML pages
- **Regex extraction** — `extract_emails()`, `extract_phones()` pull structured data
- **Normalization** — `normalize_text()` applies lowercasing, digit replacement, punctuation removal
- **Corpus statistics** — `compute_corpus_statistics()` analyzes token distributions
- **NER** — `extract_named_entities()` identifies persons, organizations, locations

**Data Flow:**
```
URL/HTML → fetch_clean_text → raw text → normalize_text → clean tokens
```

### 2. Language Models (`language_models.py`)

Implements statistical generative language models from scratch.

**Models:**
| Model | Class | Smoothing |
|-------|-------|-----------|
| Unigram | `UnigramModel` | None |
| Smoothed Unigram | `SmoothedUnigramModel` | Laplace (Add-1) |
| Bigram | `BigramModel` | None |
| Smoothed Bigram | `SmoothedBigramModelLI` | Linear Interpolation |

**Key Methods:**
- `getSentenceProbability(sentence)` — Compute P(sentence)
- `generateSentence()` — Sample a sentence from the model
- `get_perplexity(model, test_sentences)` — Evaluate model quality

**Data Flow:**
```
NLTK corpus → preprocess_corpus → train model → generate/evaluate
```

### 3. Byte Pair Encoding (`bpe.py`)

Implements subword tokenization from scratch, designed for languages like Urdu that lack clear word boundaries.

**Functions:**
- `train_bpe(wordlist, num_merges)` — Learn merge rules from vocabulary
- `apply_bpe(text, merge_rules)` — Segment text using learned rules

**Algorithm:**
1. Start with character-level vocabulary
2. Count adjacent symbol pair frequencies
3. Merge the most frequent pair
4. Repeat for `num_merges` iterations

### 4. Embeddings (`embeddings.py`)

Implements the full spectrum from sparse to dense text representations.

**Components:**

| Type | Implementation | Description |
|------|---------------|-------------|
| Sparse | `CustomTFIDF` | Term Frequency–Inverse Document Frequency |
| Count-based | `build_ppmi_matrix()` | Positive Pointwise Mutual Information |
| Neural | `train_sgns()` | Skip-Gram with Negative Sampling (PyTorch) |

**Evaluation Utilities:**
- `cosine_similarity()` — Vector similarity measurement
- `get_nearest_neighbors()` — Find semantically similar words
- `generate_training_data()` — Create (target, context) pairs for SGNS

## Data Flow

```
                    ┌─────────────┐
                    │  Raw Text   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Preprocessing│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───┐ ┌──────▼──────┐
       │Language Model│ │ BPE  │ │  Embeddings │
       └──────┬──────┘ └──┬───┘ └──────┬──────┘
              │           │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  Perplexity  │    │     │  Evaluation │
       └─────────────┘    │     └─────────────┘
                          │
                   ┌──────▼──────┐
                   │  Segmented  │
                   │    Text     │
                   └─────────────┘
```

## Dependencies Between Modules

- `preprocessing` is standalone (no internal dependencies)
- `language_models` depends on preprocessed corpus data
- `bpe` is standalone (operates on raw character sequences)
- `embeddings` depends on preprocessed corpus data
