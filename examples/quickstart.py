"""
Quickstart Example — NLP Pipeline From Scratch

Demonstrates basic usage of each module in the pipeline.
"""

from src.nlp_pipeline.preprocessing import normalize_text, extract_emails, extract_phones
from src.nlp_pipeline.language_models import preprocess_corpus, UnigramModel
from src.nlp_pipeline.bpe import train_bpe, apply_bpe
from src.nlp_pipeline.embeddings import CustomTFIDF, cosine_similarity


def main():
    # ---- 1. Text Preprocessing ----
    print("=" * 50)
    print("1. TEXT PREPROCESSING")
    print("=" * 50)

    sample_text = (
        "Contact Dr. Smith at smith@university.edu or call 555-123-4567. "
        "The 2024 NLP conference featured 15 keynote speakers."
    )

    normalized = normalize_text(sample_text)
    emails = extract_emails(sample_text)
    phones = extract_phones(sample_text)

    print(f"Original:   {sample_text}")
    print(f"Normalized: {normalized}")
    print(f"Emails:     {emails}")
    print(f"Phones:     {phones}")

    # ---- 2. Language Models ----
    print("\n" + "=" * 50)
    print("2. LANGUAGE MODELS")
    print("=" * 50)

    sentences = [
        ["The", "cat", "sat", "on", "the", "mat"],
        ["The", "dog", "ran", "in", "the", "park"],
        ["A", "bird", "flew", "over", "the", "tree"],
    ]

    processed = preprocess_corpus(sentences)
    vocab = set(token for sent in processed for token in sent)
    model = UnigramModel(processed, vocab)

    test_sent = ["<s>", "the", "cat", "</s>"]
    prob = model.getSentenceProbability(test_sent)
    generated = model.generateSentence()

    print(f"Test sentence probability: {prob:.6f}")
    print(f"Generated sentence: {' '.join(generated)}")

    # ---- 3. BPE Tokenization ----
    print("\n" + "=" * 50)
    print("3. BPE TOKENIZATION")
    print("=" * 50)

    words = ["lower", "lowest", "newer", "newest", "wider"]
    rules = train_bpe(words, num_merges=10)
    segmented = apply_bpe("lowest", rules)

    print(f"Vocabulary: {words}")
    print(f"Merge rules learned: {len(rules)}")
    print(f"Segmented 'lowest': {segmented}")

    # ---- 4. TF-IDF Embeddings ----
    print("\n" + "=" * 50)
    print("4. TF-IDF EMBEDDINGS")
    print("=" * 50)

    docs = [
        ["machine", "learning", "is", "a", "subset", "of", "ai"],
        ["deep", "learning", "uses", "neural", "networks"],
        ["natural", "language", "processing", "handles", "text"],
    ]

    tfidf = CustomTFIDF(min_freq=1)
    tfidf.fit(docs)
    matrix = tfidf.transform(docs)

    sim = cosine_similarity(matrix[0], matrix[1])
    print(f"Vocabulary size: {tfidf.vocab_size}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Cosine similarity (doc 1 vs doc 2): {sim:.4f}")

    print("\n✅ All modules working correctly!")


if __name__ == "__main__":
    main()
