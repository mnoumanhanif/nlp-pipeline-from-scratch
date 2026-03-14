# Setup Guide

This guide walks you through setting up the NLP Pipeline project on your local machine.

## Prerequisites

- **Python 3.8+** — [Download Python](https://www.python.org/downloads/)
- **pip** — Comes with Python (ensure it is up to date with `pip install --upgrade pip`)
- **Git** — [Download Git](https://git-scm.com/downloads)
- (Optional) **CUDA-compatible GPU** — Speeds up neural embedding training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/nlp-pipeline-from-scratch.git
cd nlp-pipeline-from-scratch
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('brown')
nltk.download('reuters')
```

### 5. Verify Installation

```bash
pytest tests/ -v
```

## Running the Notebook

The original interactive notebook is preserved in the `notebooks/` directory:

```bash
jupyter notebook notebooks/NLP_Assignment_01_\(24K_8001\).ipynb
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated and dependencies installed |
| NLTK download errors | Run the NLTK download commands above or check network connection |
| GPU not detected | Verify CUDA installation; CPU fallback works automatically |
| Import errors from `src/` | Run from the project root directory |
