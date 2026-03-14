"""
Word embedding models: TF-IDF, PPMI, and Skip-Gram with Negative Sampling (SGNS).

Implements sparse and dense text representations from scratch, including
custom TF-IDF, count-based PPMI embeddings, and a neural Skip-Gram model
built natively in PyTorch.
"""

import math
import random
from collections import Counter, defaultdict

import numpy as np


class CustomTFIDF:
    """Custom TF-IDF implementation from scratch."""

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.vocab = {}
        self.idf = {}
        self.vocab_size = 0

    def fit(self, documents: list) -> None:
        """
        Build vocabulary and compute IDF from training documents.

        Args:
            documents: List of documents, each a list of tokens.
        """
        doc_counts = Counter()
        for doc in documents:
            doc_counts.update(set(doc))

        filtered_words = [
            word for word, count in doc_counts.items() if count >= self.min_freq
        ]
        self.vocab = {word: idx for idx, word in enumerate(filtered_words)}
        self.vocab_size = len(self.vocab)

        num_docs = len(documents)
        for word in self.vocab:
            df = doc_counts[word]
            self.idf[word] = math.log(num_docs / (df + 1))

    def transform(self, documents: list) -> np.ndarray:
        """
        Transform documents into TF-IDF matrix.

        Args:
            documents: List of documents, each a list of tokens.

        Returns:
            TF-IDF matrix of shape (num_documents, vocab_size).
        """
        num_docs = len(documents)
        tfidf_matrix = np.zeros((num_docs, self.vocab_size))

        for i, doc in enumerate(documents):
            word_counts = Counter(doc)
            total_words = len(doc)

            for word, count in word_counts.items():
                if word in self.vocab:
                    tf = count / total_words
                    tfidf_matrix[i, self.vocab[word]] = tf * self.idf[word]

        return tfidf_matrix


def build_ppmi_matrix(
    docs: list, vocab: dict, vocab_size: int, window_size: int = 2
) -> np.ndarray:
    """
    Build a Positive Pointwise Mutual Information (PPMI) matrix.

    Args:
        docs: List of documents, each a list of tokens.
        vocab: Dictionary mapping words to indices.
        vocab_size: Size of the vocabulary.
        window_size: Context window size for co-occurrence counting.

    Returns:
        PPMI matrix of shape (vocab_size, vocab_size).
    """
    co_counts = defaultdict(float)
    w_counts = defaultdict(float)
    c_counts = defaultdict(float)
    total_pairs = 0

    for doc in docs:
        n = len(doc)
        for i, target in enumerate(doc):
            if target not in vocab:
                continue
            target_idx = vocab[target]

            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context = doc[j]
                if context not in vocab:
                    continue
                context_idx = vocab[context]

                co_counts[(target_idx, context_idx)] += 1
                w_counts[target_idx] += 1
                c_counts[context_idx] += 1
                total_pairs += 1

    ppmi = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for (w, c), count in co_counts.items():
        p_wc = count / total_pairs
        p_w = w_counts[w] / total_pairs
        p_c = c_counts[c] / total_pairs

        pmi = math.log2(p_wc / (p_w * p_c))
        ppmi[w, c] = max(0, pmi)

    return ppmi


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def get_nearest_neighbors(
    matrix: np.ndarray, vocab: dict, idx_to_word: dict,
    word: str, top_k: int = 5
) -> list:
    """
    Find the nearest neighbors of a word in an embedding space.

    Args:
        matrix: Embedding matrix (vocab_size x embedding_dim).
        vocab: Dictionary mapping words to indices.
        idx_to_word: Dictionary mapping indices to words.
        word: Query word.
        top_k: Number of neighbors to return.

    Returns:
        List of (word, similarity) tuples.
    """
    if word not in vocab:
        return []

    w_idx = vocab[word]
    w_vec = matrix[w_idx]

    if np.linalg.norm(w_vec) == 0:
        return []

    norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.dot(matrix, w_vec) / (norms * np.linalg.norm(w_vec))
        sims = np.nan_to_num(sims)

    nearest_indices = np.argsort(sims)[-(top_k + 1):][::-1]
    return [
        (idx_to_word[idx], float(sims[idx]))
        for idx in nearest_indices
        if idx != w_idx
    ][:top_k]


def generate_training_data(
    docs: list, vocab: dict, window_size: int = 2
) -> list:
    """
    Generate (target, context) training pairs for Skip-Gram.

    Args:
        docs: List of documents, each a list of tokens.
        vocab: Dictionary mapping words to indices.
        window_size: Context window size.

    Returns:
        List of (target_idx, context_idx) tuples.
    """
    pairs = []
    for doc in docs:
        indices = [vocab[w] for w in doc if w in vocab]
        for i, target in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((target, indices[j]))
    return pairs


def train_sgns(
    pairs: list, vocab_size: int, embed_size: int = 50,
    epochs: int = 3, batch_size: int = 2048, num_negatives: int = 5
) -> np.ndarray:
    """
    Train Skip-Gram with Negative Sampling using PyTorch.

    Args:
        pairs: List of (target_idx, context_idx) training pairs.
        vocab_size: Size of the vocabulary.
        embed_size: Dimensionality of embeddings.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        num_negatives: Number of negative samples per positive pair.

    Returns:
        Trained embedding matrix as numpy array (vocab_size x embed_size).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SkipGramNegativeSampling(nn.Module):
        def __init__(self, vocab_size, embed_size):
            super().__init__()
            self.target_embeddings = nn.Embedding(vocab_size, embed_size)
            self.context_embeddings = nn.Embedding(vocab_size, embed_size)

            initrange = 0.5 / embed_size
            self.target_embeddings.weight.data.uniform_(-initrange, initrange)
            self.context_embeddings.weight.data.uniform_(-initrange, initrange)

        def forward(self, target, context, negatives):
            emb_target = self.target_embeddings(target)
            emb_context = self.context_embeddings(context)
            emb_negatives = self.context_embeddings(negatives)

            pos_score = torch.sum(emb_target * emb_context, dim=1)
            pos_loss = -torch.nn.functional.logsigmoid(pos_score)

            neg_score = torch.bmm(
                emb_negatives, emb_target.unsqueeze(2)
            ).squeeze(2)
            neg_loss = -torch.sum(
                torch.nn.functional.logsigmoid(-neg_score), dim=1
            )

            return torch.mean(pos_loss + neg_loss)

    model = SkipGramNegativeSampling(vocab_size, embed_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i: i + batch_size]
            targets = torch.tensor(
                [p[0] for p in batch_pairs], dtype=torch.long
            ).to(device)
            contexts = torch.tensor(
                [p[1] for p in batch_pairs], dtype=torch.long
            ).to(device)
            negatives = torch.randint(
                0, vocab_size, (len(batch_pairs), num_negatives),
                dtype=torch.long,
            ).to(device)

            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model.target_embeddings.weight.data.cpu().numpy()
