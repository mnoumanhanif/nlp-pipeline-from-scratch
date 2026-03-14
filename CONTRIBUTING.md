# Contributing to NLP Pipeline From Scratch

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/nlp-pipeline-from-scratch.git
   cd nlp-pipeline-from-scratch
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes in the `src/` directory
2. Add or update tests in `tests/`
3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
4. Run the linter:
   ```bash
   flake8 src/ tests/ --max-line-length=100
   ```
5. Commit your changes with a clear message:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
6. Push and open a pull request

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused

## Pull Request Process

1. Update documentation if your changes affect the public API
2. Add tests for new functionality
3. Ensure all tests pass before submitting
4. Use a descriptive PR title and fill out the PR template
5. Request a review from a maintainer

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) to report issues.

## Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) for suggestions.

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on the technical merits of contributions
- Help newcomers feel welcome

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
