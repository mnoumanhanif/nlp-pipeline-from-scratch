"""
Byte Pair Encoding (BPE) implementation from scratch.

Designed for subword tokenization of languages without clear word boundaries,
such as Urdu.
"""

import collections


def train_bpe(wordlist: list, num_merges: int = 3000) -> list:
    """
    Train BPE merge rules from a word list.

    Iteratively merges the most frequent adjacent character pairs to learn
    subword units from a vocabulary.

    Args:
        wordlist: List of words to train BPE on.
        num_merges: Maximum number of merge operations to learn.

    Returns:
        List of merge rules as (symbol1, symbol2) tuples.
    """
    vocab = {
        " ".join(list(w.strip())): 1 for w in set(wordlist) if w.strip()
    }
    merge_rules = []

    for _ in range(num_merges):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pairs[symbols[j], symbols[j + 1]] += freq

        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merge_rules.append(best_pair)

        new_vocab = {}
        for word, freq in vocab.items():
            symbols = word.split()
            new_symbols = []
            j = 0
            while j < len(symbols):
                if (
                    j < len(symbols) - 1
                    and symbols[j] == best_pair[0]
                    and symbols[j + 1] == best_pair[1]
                ):
                    new_symbols.append(best_pair[0] + best_pair[1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1
            new_vocab[" ".join(new_symbols)] = freq
        vocab = new_vocab

    return merge_rules


def apply_bpe(text: str, merge_rules: list) -> str:
    """
    Apply learned BPE merge rules to segment text.

    Args:
        text: Input text (can be unspaced).
        merge_rules: List of merge rules from train_bpe.

    Returns:
        Segmented text with spaces between subword units.
    """
    symbols = list(text.strip())

    for pair in merge_rules:
        if pair[0] not in symbols or pair[1] not in symbols:
            continue

        new_symbols = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    return " ".join(symbols)
