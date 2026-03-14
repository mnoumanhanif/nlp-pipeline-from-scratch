# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-03-14

### Added
- Modular project structure with `src/nlp_pipeline/` package
- `preprocessing.py` — text extraction, normalization, regex, NER utilities
- `language_models.py` — unigram, bigram, smoothed models with perplexity evaluation
- `bpe.py` — Byte Pair Encoding training and application from scratch
- `embeddings.py` — TF-IDF, PPMI, and Skip-Gram SGNS implementations
- Comprehensive test suite in `tests/`
- Documentation in `docs/` (setup, architecture, development guides)
- GitHub Actions CI workflow
- Issue and pull request templates
- `CONTRIBUTING.md` with contribution guidelines
- `requirements.txt` with pinned dependency ranges
- `.gitignore` for Python projects

### Changed
- Reorganized repository from single notebook to modular package structure
- Moved data files to `data/` directory
- Moved notebook to `notebooks/` directory
- Rewrote `README.md` with comprehensive project documentation

### Removed
- PDF export of notebook (binary file not suitable for version control)
