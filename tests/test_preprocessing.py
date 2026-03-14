"""Tests for the preprocessing module."""

from src.nlp_pipeline.preprocessing import (
    extract_emails,
    extract_phones,
    normalize_text,
    compute_corpus_statistics,
)


class TestExtractEmails:
    def test_extracts_standard_emails(self):
        text = "Contact us at hello@example.com or support@test.org"
        emails = extract_emails(text)
        assert "hello@example.com" in emails
        assert "support@test.org" in emails

    def test_extracts_uppercase_emails(self):
        text = "Email RMM688@MIT.EDU for info"
        emails = extract_emails(text)
        assert "RMM688@MIT.EDU" in emails

    def test_returns_empty_for_no_emails(self):
        text = "No emails here, just regular text."
        assert extract_emails(text) == []

    def test_deduplicates_emails(self):
        text = "Email test@example.com and test@example.com again"
        emails = extract_emails(text)
        assert len(emails) == 1


class TestExtractPhones:
    def test_extracts_us_phone_numbers(self):
        text = "Call us at 555-123-4567"
        phones = extract_phones(text)
        assert len(phones) == 1

    def test_extracts_local_numbers(self):
        text = "Local: 253-5856"
        phones = extract_phones(text)
        assert len(phones) == 1

    def test_returns_empty_for_no_phones(self):
        text = "No phone numbers here."
        assert extract_phones(text) == []


class TestNormalizeText:
    def test_lowercases_text(self):
        assert "hello world" in normalize_text("Hello World")

    def test_replaces_digits(self):
        result = normalize_text("There are 42 items")
        # <NUM> markers have angle brackets stripped by punctuation removal
        assert "NUM" in result
        assert "42" not in result

    def test_removes_punctuation(self):
        result = normalize_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_removes_extra_whitespace(self):
        result = normalize_text("hello   world")
        assert "  " not in result

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestCorpusStatistics:
    def test_basic_statistics(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        stats = compute_corpus_statistics(tokens)
        assert stats["total_tokens"] == 6
        assert stats["vocab_size"] == 5
        assert len(stats["top_25"]) == 5

    def test_with_stopwords(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        stop_words = {"the", "on"}
        stats = compute_corpus_statistics(tokens, stop_words)
        assert stats["filtered_tokens"] == 3
        assert "filtered_top_25" in stats

    def test_empty_tokens(self):
        stats = compute_corpus_statistics([])
        assert stats["total_tokens"] == 0
        assert stats["vocab_size"] == 0
        assert stats["ttr"] == 0
