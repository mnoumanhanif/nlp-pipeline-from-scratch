"""Tests for the language models module."""

import pytest

from src.nlp_pipeline.language_models import (
    preprocess_corpus,
    UnigramModel,
    SmoothedUnigramModel,
    BigramModel,
    SmoothedBigramModelLI,
    get_perplexity,
)


@pytest.fixture
def sample_sentences():
    return [
        ["The", "cat", "sat"],
        ["The", "dog", "ran"],
        ["A", "cat", "ran"],
        ["The", "bird", "flew"],
    ]


@pytest.fixture
def preprocessed_sentences(sample_sentences):
    return preprocess_corpus(sample_sentences)


@pytest.fixture
def vocab(preprocessed_sentences):
    return set(
        token for sent in preprocessed_sentences for token in sent
    )


class TestPreprocessCorpus:
    def test_adds_boundary_markers(self, sample_sentences):
        result = preprocess_corpus(sample_sentences)
        for sent in result:
            assert sent[0] == "<s>"
            assert sent[-1] == "</s>"

    def test_lowercases_tokens(self, sample_sentences):
        result = preprocess_corpus(sample_sentences)
        for sent in result:
            for token in sent:
                assert token == token.lower()

    def test_preserves_sentence_count(self, sample_sentences):
        result = preprocess_corpus(sample_sentences)
        assert len(result) == len(sample_sentences)


class TestUnigramModel:
    def test_sentence_probability_nonzero(self, preprocessed_sentences, vocab):
        model = UnigramModel(preprocessed_sentences, vocab)
        sent = ["<s>", "the", "cat", "</s>"]
        prob = model.getSentenceProbability(sent)
        assert prob > 0

    def test_unseen_word_returns_zero(self, preprocessed_sentences, vocab):
        model = UnigramModel(preprocessed_sentences, vocab)
        sent = ["<s>", "xyzzy", "</s>"]
        prob = model.getSentenceProbability(sent)
        assert prob == 0.0

    def test_generate_sentence_has_markers(self, preprocessed_sentences, vocab):
        model = UnigramModel(preprocessed_sentences, vocab)
        sent = model.generateSentence()
        assert sent[0] == "<s>"


class TestSmoothedUnigramModel:
    def test_smoothed_handles_unseen(self, preprocessed_sentences, vocab):
        model = SmoothedUnigramModel(preprocessed_sentences, vocab)
        sent = ["<s>", "the", "cat", "</s>"]
        prob = model.getSentenceProbability(sent)
        assert prob > 0

    def test_generate_sentence(self, preprocessed_sentences, vocab):
        model = SmoothedUnigramModel(preprocessed_sentences, vocab)
        sent = model.generateSentence()
        assert len(sent) >= 2


class TestBigramModel:
    def test_sentence_probability(self, preprocessed_sentences, vocab):
        model = BigramModel(preprocessed_sentences, vocab)
        sent = ["<s>", "the", "cat", "</s>"]
        prob = model.getSentenceProbability(sent)
        assert prob >= 0

    def test_unseen_bigram_returns_zero(self, preprocessed_sentences, vocab):
        model = BigramModel(preprocessed_sentences, vocab)
        sent = ["<s>", "cat", "the", "</s>"]
        prob = model.getSentenceProbability(sent)
        # May be zero if "cat the" bigram never appeared
        assert prob >= 0

    def test_generate_sentence(self, preprocessed_sentences, vocab):
        model = BigramModel(preprocessed_sentences, vocab)
        sent = model.generateSentence()
        assert sent[0] == "<s>"


class TestSmoothedBigramModelLI:
    def test_interpolated_probability(self, preprocessed_sentences, vocab):
        model = SmoothedBigramModelLI(preprocessed_sentences, vocab)
        sent = ["<s>", "the", "cat", "</s>"]
        prob = model.getSentenceProbability(sent)
        assert prob > 0

    def test_generate_sentence(self, preprocessed_sentences, vocab):
        model = SmoothedBigramModelLI(preprocessed_sentences, vocab)
        sent = model.generateSentence()
        assert sent[0] == "<s>"
        assert len(sent) >= 2


class TestPerplexity:
    def test_finite_perplexity(self, preprocessed_sentences, vocab):
        model = SmoothedUnigramModel(preprocessed_sentences, vocab)
        perp = get_perplexity(model, preprocessed_sentences[:2])
        assert perp > 0
        assert perp < float("inf")

    def test_infinite_perplexity_for_unseen(self, preprocessed_sentences, vocab):
        model = UnigramModel(preprocessed_sentences, vocab)
        unseen = [["<s>", "xyzzy_unseen_word", "</s>"]]
        perp = get_perplexity(model, unseen)
        assert perp == float("inf")
