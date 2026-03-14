"""Tests for the BPE module."""

from src.nlp_pipeline.bpe import train_bpe, apply_bpe


class TestTrainBPE:
    def test_returns_merge_rules(self):
        wordlist = ["hello", "world", "help", "held"]
        rules = train_bpe(wordlist, num_merges=5)
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_merge_rules_are_tuples(self):
        wordlist = ["abc", "abd", "abx"]
        rules = train_bpe(wordlist, num_merges=3)
        for rule in rules:
            assert isinstance(rule, tuple)
            assert len(rule) == 2

    def test_empty_wordlist(self):
        rules = train_bpe([], num_merges=10)
        assert rules == []

    def test_single_char_words(self):
        wordlist = ["a", "b", "c"]
        rules = train_bpe(wordlist, num_merges=5)
        assert rules == []

    def test_num_merges_limit(self):
        wordlist = ["hello", "world", "help", "held", "hero"]
        rules = train_bpe(wordlist, num_merges=2)
        assert len(rules) <= 2


class TestApplyBPE:
    def test_basic_segmentation(self):
        wordlist = ["hello", "help", "held"]
        rules = train_bpe(wordlist, num_merges=10)
        result = apply_bpe("hello", rules)
        assert isinstance(result, str)
        # Result should contain the characters of "hello"
        assert result.replace(" ", "") == "hello"

    def test_empty_text(self):
        rules = [("a", "b")]
        result = apply_bpe("", rules)
        assert result == ""

    def test_no_rules(self):
        result = apply_bpe("abc", [])
        assert result == "a b c"

    def test_single_character(self):
        result = apply_bpe("a", [])
        assert result == "a"
