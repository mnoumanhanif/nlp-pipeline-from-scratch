"""
Statistical N-gram language models built from scratch.

Includes unigram and bigram models with smoothing techniques
(Laplace and Linear Interpolation), perplexity evaluation,
and sentence generation.
"""

import math
import random
from collections import Counter, defaultdict


def preprocess_corpus(sentences: list) -> list:
    """
    Adds <s> and </s> markers to each sentence and lowercases tokens.

    Args:
        sentences: List of sentences, where each sentence is a list of tokens.

    Returns:
        List of preprocessed sentences with boundary markers.
    """
    processed = []
    for sentence in sentences:
        processed_sent = ["<s>"] + [word.lower() for word in sentence] + ["</s>"]
        processed.append(processed_sent)
    return processed


class UnigramModel:
    """Unsmoothed unigram language model."""

    def __init__(self, train_sentences: list, vocab: set):
        self.vocab = list(vocab)
        self.counts = Counter(
            token for sent in train_sentences for token in sent
        )
        self.total_tokens = sum(self.counts.values())
        self.probs = [self.counts[w] / self.total_tokens for w in self.vocab]

    def getSentenceProbability(self, sen: list) -> float:
        """Calculate the probability of a sentence."""
        log_prob = 0
        for word in sen:
            if word == "<s>":
                continue
            word_count = self.counts.get(word, 0)
            if word_count == 0:
                return 0.0
            log_prob += math.log(word_count / self.total_tokens)
        return math.exp(log_prob)

    def generateSentence(self) -> list:
        """Generate a random sentence from the model."""
        sentence = ["<s>"]
        current_word = None
        while current_word != "</s>":
            current_word = random.choices(self.vocab, weights=self.probs)[0]
            if current_word == "<s>":
                continue
            sentence.append(current_word)
            if len(sentence) > 100:
                break
        return sentence


class SmoothedUnigramModel:
    """Unigram language model with Laplace (Add-1) smoothing."""

    def __init__(self, train_sentences: list, vocab: set):
        self.vocab = list(vocab)
        self.counts = Counter(
            token for sent in train_sentences for token in sent
        )
        self.total_tokens = sum(self.counts.values())
        self.vocab_size = len(self.vocab)
        self.probs = [
            (self.counts[w] + 1) / (self.total_tokens + self.vocab_size)
            for w in self.vocab
        ]

    def getSentenceProbability(self, sen: list) -> float:
        """Calculate the smoothed probability of a sentence."""
        log_prob = 0
        for word in sen:
            if word == "<s>":
                continue
            word_count = self.counts.get(word, 0)
            log_prob += math.log(
                (word_count + 1) / (self.total_tokens + self.vocab_size)
            )
        return math.exp(log_prob)

    def generateSentence(self) -> list:
        """Generate a random sentence from the smoothed model."""
        sentence = ["<s>"]
        current_word = None
        while current_word != "</s>":
            current_word = random.choices(self.vocab, weights=self.probs)[0]
            if current_word == "<s>":
                continue
            sentence.append(current_word)
            if len(sentence) > 100:
                break
        return sentence


class BigramModel:
    """Unsmoothed bigram language model."""

    def __init__(self, train_sentences: list, vocab: set):
        self.vocab = list(vocab)
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()

        for sent in train_sentences:
            for i in range(len(sent)):
                self.unigram_counts[sent[i]] += 1
                if i < len(sent) - 1:
                    self.bigram_counts[(sent[i], sent[i + 1])] += 1

        self.transitions = defaultdict(lambda: ([], []))
        for (w1, w2), count in self.bigram_counts.items():
            self.transitions[w1][0].append(w2)
            self.transitions[w1][1].append(count / self.unigram_counts[w1])

    def getSentenceProbability(self, sen: list) -> float:
        """Calculate the probability of a sentence."""
        log_prob = 0
        for i in range(len(sen) - 1):
            w1, w2 = sen[i], sen[i + 1]
            count_w1 = self.unigram_counts.get(w1, 0)
            count_w1_w2 = self.bigram_counts.get((w1, w2), 0)
            if count_w1 == 0 or count_w1_w2 == 0:
                return 0.0
            log_prob += math.log(count_w1_w2 / count_w1)
        return math.exp(log_prob)

    def generateSentence(self) -> list:
        """Generate a random sentence from the bigram model."""
        sentence = ["<s>"]
        current_word = "<s>"
        while current_word != "</s>":
            next_words, probs = self.transitions.get(current_word, ([], []))
            if not next_words:
                sentence.append("</s>")
                break
            current_word = random.choices(next_words, weights=probs)[0]
            if current_word == "<s>":
                continue
            sentence.append(current_word)
            if len(sentence) > 100:
                break
        return sentence


class SmoothedBigramModelLI:
    """Bigram language model with Linear Interpolation smoothing."""

    def __init__(
        self, train_sentences: list, vocab: set, l1: float = 0.70,
        l2: float = 0.25, l3: float = 0.05
    ):
        self.vocab = list(vocab)
        self.vocab_size = len(vocab)
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.total_tokens = 0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        for sent in train_sentences:
            for i in range(len(sent)):
                self.unigram_counts[sent[i]] += 1
                self.total_tokens += 1
                if i < len(sent) - 1:
                    self.bigram_counts[(sent[i], sent[i + 1])] += 1

        self.unigram_words = list(self.unigram_counts.keys())
        self.unigram_probs = [
            self.unigram_counts[w] / self.total_tokens for w in self.unigram_words
        ]

    def getSentenceProbability(self, sen: list) -> float:
        """Calculate the interpolated probability of a sentence."""
        log_prob = 0
        for i in range(len(sen) - 1):
            w1, w2 = sen[i], sen[i + 1]

            p_bigram = 0.0
            if self.unigram_counts.get(w1, 0) > 0:
                p_bigram = (
                    self.bigram_counts.get((w1, w2), 0) / self.unigram_counts[w1]
                )

            p_unigram = self.unigram_counts.get(w2, 0) / self.total_tokens
            p_uniform = 1.0 / self.vocab_size

            p_interp = (
                (self.l1 * p_bigram)
                + (self.l2 * p_unigram)
                + (self.l3 * p_uniform)
            )
            log_prob += math.log(p_interp)
        return math.exp(log_prob)

    def generateSentence(self) -> list:
        """Generate a random sentence from the interpolated model."""
        sentence = ["<s>"]
        current_word = "<s>"
        while current_word != "</s>":
            rand_val = random.random()
            if (
                rand_val < self.l1
                and self.unigram_counts.get(current_word, 0) > 0
            ):
                next_words = [
                    pair[1]
                    for pair in self.bigram_counts.keys()
                    if pair[0] == current_word
                ]
                if next_words:
                    weights = [
                        self.bigram_counts[(current_word, nw)] for nw in next_words
                    ]
                    current_word = random.choices(next_words, weights=weights)[0]
                else:
                    current_word = random.choices(
                        self.unigram_words, weights=self.unigram_probs
                    )[0]
            elif rand_val < (self.l1 + self.l2):
                current_word = random.choices(
                    self.unigram_words, weights=self.unigram_probs
                )[0]
            else:
                current_word = random.choice(self.vocab)

            if current_word == "<s>":
                continue
            sentence.append(current_word)
            if len(sentence) > 100:
                break
        return sentence


def get_perplexity(model, test_sentences: list) -> float:
    """
    Calculate the perplexity of a language model on test sentences.

    Args:
        model: A language model with a getSentenceProbability method.
        test_sentences: List of tokenized test sentences.

    Returns:
        Perplexity score (lower is better). Returns inf for zero-probability.
    """
    log_prob_sum = 0
    total_words = 0

    for sent in test_sentences:
        total_words += len(sent) - 1
        prob = model.getSentenceProbability(sent)
        if prob == 0.0:
            return float("inf")
        log_prob_sum += math.log(prob)

    return math.exp(-log_prob_sum / total_words)


def generate_sentences_to_file(
    model, filename: str, num_sentences: int = 20
) -> None:
    """
    Generate sentences from a model and write them to a file.

    Args:
        model: A language model with a generateSentence method.
        filename: Output file path.
        num_sentences: Number of sentences to generate.
    """
    with open(filename, "w") as f:
        for _ in range(num_sentences):
            sentence = model.generateSentence()
            f.write(" ".join(sentence) + "\n")
