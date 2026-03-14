# Development Guide

This guide describes how to develop, test, and contribute to the NLP Pipeline project.

## Project Structure

```
nlp-pipeline-from-scratch/
├── src/nlp_pipeline/        # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py     # Text processing utilities
│   ├── language_models.py   # N-gram language models
│   ├── bpe.py               # Byte Pair Encoding
│   └── embeddings.py        # TF-IDF, PPMI, SGNS
├── data/                    # Data files
│   ├── wordlist.txt         # Urdu word vocabulary
│   └── 50_nospaces.txt      # Urdu test sentences
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Usage examples
└── .github/                 # CI/CD and templates
```

## Development Setup

```bash
# Clone and setup
git clone https://github.com/mnoumanhanif/nlp-pipeline-from-scratch.git
cd nlp-pipeline-from-scratch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ -v --tb=short
```

## Linting

```bash
# Run flake8
flake8 src/ tests/ --max-line-length=100 --ignore=E501
```

## Code Style

- Follow **PEP 8** conventions
- Use **type hints** for function signatures
- Write **docstrings** for all public functions and classes
- Keep functions focused — each should do one thing well
- Use descriptive variable names

## Adding a New Module

1. Create the module in `src/nlp_pipeline/`
2. Add corresponding tests in `tests/`
3. Update documentation in `docs/`
4. Run the full test suite to verify

## Jupyter Notebook

The original notebook in `notebooks/` demonstrates the full pipeline interactively. It uses modules from `src/nlp_pipeline/` and is useful for exploration and visualization.

To run:

```bash
jupyter notebook notebooks/
```

## Continuous Integration

GitHub Actions runs on every push and pull request:

- Linting with flake8
- Full test suite with pytest
- Python 3.9, 3.10, and 3.11 compatibility
