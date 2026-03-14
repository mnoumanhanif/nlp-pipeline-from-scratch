"""Tests for the embeddings module."""

import numpy as np
import pytest

from src.nlp_pipeline.embeddings import (
    CustomTFIDF,
    cosine_similarity,
    get_nearest_neighbors,
    generate_training_data,
)


class TestCustomTFIDF:
    @pytest.fixture
    def sample_docs(self):
        return [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "chased", "the", "cat"],
            ["a", "bird", "flew", "over", "the", "mat"],
        ]

    def test_fit_builds_vocabulary(self, sample_docs):
        tfidf = CustomTFIDF(min_freq=1)
        tfidf.fit(sample_docs)
        assert tfidf.vocab_size > 0

    def test_transform_returns_correct_shape(self, sample_docs):
        tfidf = CustomTFIDF(min_freq=1)
        tfidf.fit(sample_docs)
        matrix = tfidf.transform(sample_docs)
        assert matrix.shape == (3, tfidf.vocab_size)

    def test_min_freq_filters_rare_words(self, sample_docs):
        tfidf_low = CustomTFIDF(min_freq=1)
        tfidf_low.fit(sample_docs)

        tfidf_high = CustomTFIDF(min_freq=3)
        tfidf_high.fit(sample_docs)

        assert tfidf_high.vocab_size <= tfidf_low.vocab_size

    def test_tfidf_values_are_finite(self, sample_docs):
        tfidf = CustomTFIDF(min_freq=1)
        tfidf.fit(sample_docs)
        matrix = tfidf.transform(sample_docs)
        assert np.all(np.isfinite(matrix))


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert cosine_similarity(v1, v2) == 0.0


class TestGetNearestNeighbors:
    def test_finds_neighbors(self):
        matrix = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])
        vocab = {"a": 0, "b": 1, "c": 2}
        idx_to_word = {0: "a", 1: "b", 2: "c"}

        neighbors = get_nearest_neighbors(matrix, vocab, idx_to_word, "a", top_k=2)
        assert len(neighbors) == 2
        assert neighbors[0][0] == "b"  # Most similar to "a"

    def test_oov_word(self):
        matrix = np.array([[1.0, 0.0]])
        vocab = {"a": 0}
        idx_to_word = {0: "a"}

        neighbors = get_nearest_neighbors(
            matrix, vocab, idx_to_word, "unknown", top_k=1
        )
        assert neighbors == []


class TestGenerateTrainingData:
    def test_generates_pairs(self):
        docs = [["a", "b", "c", "d"]]
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        pairs = generate_training_data(docs, vocab, window_size=1)
        assert len(pairs) > 0
        for target, context in pairs:
            assert 0 <= target < 4
            assert 0 <= context < 4

    def test_window_size(self):
        docs = [["a", "b", "c", "d", "e"]]
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

        pairs_small = generate_training_data(docs, vocab, window_size=1)
        pairs_large = generate_training_data(docs, vocab, window_size=2)

        assert len(pairs_large) > len(pairs_small)

    def test_empty_docs(self):
        pairs = generate_training_data([], {"a": 0}, window_size=1)
        assert pairs == []
