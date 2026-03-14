# NLP Pipeline From Scratch 🚀

A comprehensive, end-to-end Natural Language Processing system built from scratch — bridging classical statistical methods and modern deep learning.

[![CI](https://github.com/mnoumanhanif/nlp-pipeline-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/mnoumanhanif/nlp-pipeline-from-scratch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📌 Overview

To truly master Large Language Models (LLMs), one must understand the foundational mechanics that power them. This repository implements every layer of an NLP pipeline from the ground up — from raw text acquisition to neural word embeddings — demonstrating the progression from classical to modern NLP techniques.

## ✨ Key Features

- **Text Preprocessing** — Web scraping, regex extraction, normalization, POS tagging, NER
- **N-Gram Language Models** — Unigram and bigram models with Laplace and Linear Interpolation smoothing
- **Byte Pair Encoding (BPE)** — Subword tokenization for Urdu from scratch
- **Word Embeddings** — TF-IDF, PPMI, and Skip-Gram with Negative Sampling (PyTorch)
- **Evaluation** — Intrinsic (word similarity) and extrinsic (topic classification) metrics
- **Modular Design** — Clean, reusable Python modules with full test coverage

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| NLP | NLTK, Hugging Face Tokenizers |
| ML | Scikit-learn, NumPy, SciPy |
| Data | Hugging Face Datasets, BeautifulSoup |
| Testing | pytest |
| CI/CD | GitHub Actions |

## 📁 Project Structure

```
nlp-pipeline-from-scratch/
├── src/nlp_pipeline/          # Core source modules
│   ├── __init__.py
│   ├── preprocessing.py       # Text processing, regex, NER
│   ├── language_models.py     # N-gram models and perplexity
│   ├── bpe.py                 # Byte Pair Encoding
│   └── embeddings.py          # TF-IDF, PPMI, SGNS
├── data/                      # Dataset files
│   ├── wordlist.txt           # Urdu vocabulary (5,690 words)
│   └── 50_nospaces.txt        # Urdu test sentences
├── notebooks/                 # Interactive Jupyter notebooks
├── tests/                     # Automated test suite
├── docs/                      # Documentation
│   ├── setup.md               # Installation guide
│   ├── architecture.md        # System architecture
│   └── development.md         # Development guide
├── examples/                  # Usage examples
│   └── quickstart.py          # Getting started script
├── .github/                   # CI/CD and templates
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── requirements.txt           # Python dependencies
├── CONTRIBUTING.md            # Contribution guidelines
├── CHANGELOG.md               # Version history
├── LICENSE                    # Apache 2.0
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/nlp-pipeline-from-scratch.git
cd nlp-pipeline-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
            'maxent_ne_chunker_tab', 'words', 'wordnet', 'omw-1.4']:
    nltk.download(pkg)
"
```

## 📖 Usage

### Quick Start

```bash
python examples/quickstart.py
```

### Using Individual Modules

**Text Preprocessing:**
```python
from src.nlp_pipeline.preprocessing import normalize_text, extract_emails

text = "Contact Dr. Smith at smith@mit.edu. The 2024 conference had 15 talks."
print(normalize_text(text))
# Output: "contact dr smith at smithmitedu the <NUM> conference had <NUM> talks"

print(extract_emails(text))
# Output: ['smith@mit.edu']
```

**Language Models:**
```python
from src.nlp_pipeline.language_models import preprocess_corpus, UnigramModel

sentences = [["The", "cat", "sat"], ["The", "dog", "ran"]]
processed = preprocess_corpus(sentences)
vocab = set(token for sent in processed for token in sent)

model = UnigramModel(processed, vocab)
print(" ".join(model.generateSentence()))
```

**BPE Tokenization:**
```python
from src.nlp_pipeline.bpe import train_bpe, apply_bpe

words = ["lower", "lowest", "newer", "newest"]
rules = train_bpe(words, num_merges=10)
print(apply_bpe("lowest", rules))
```

**TF-IDF Embeddings:**
```python
from src.nlp_pipeline.embeddings import CustomTFIDF, cosine_similarity

docs = [["machine", "learning"], ["deep", "learning"], ["natural", "language"]]
tfidf = CustomTFIDF(min_freq=1)
tfidf.fit(docs)
matrix = tfidf.transform(docs)
print(f"Similarity: {cosine_similarity(matrix[0], matrix[1]):.4f}")
```

### Interactive Notebook

For the full interactive pipeline experience:

```bash
jupyter notebook notebooks/NLP_Assignment_01_\(24K_8001\).ipynb
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_preprocessing.py -v
pytest tests/test_language_models.py -v
pytest tests/test_bpe.py -v
pytest tests/test_embeddings.py -v
```

## 🔬 Technical Details

### 1. Text Preprocessing Pipeline

Web scraping with **BeautifulSoup**, regex-based information extraction (emails, phone numbers), text normalization, and linguistic annotation using **NLTK** (POS tagging, lemmatization, NER).

### 2. Statistical Language Models

**Unigram** and **Bigram** models implemented from scratch with **Laplace smoothing** and **Linear Interpolation**. Evaluated using **perplexity** on in-domain (Brown corpus) and out-of-domain (Reuters) data.

### 3. Subword Tokenization

**Byte Pair Encoding (BPE)** implemented from scratch for segmenting **Urdu text** — a language without clear word boundaries. The algorithm learns statistical co-occurrence patterns to identify subword units.

### 4. Word Embeddings

Progression from sparse to dense representations:
- **TF-IDF** — Custom from-scratch implementation
- **PPMI** — Co-occurrence based count embeddings
- **SGNS** — Skip-Gram with Negative Sampling neural model in **PyTorch**

Evaluated with **Spearman correlation** (intrinsic) and **topic classification** on AG News (extrinsic).

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhammad Nouman Hanif** — [@mnoumanhanif](https://github.com/mnoumanhanif)

---

> *"To understand LLMs, build the pipeline they're built on."*
