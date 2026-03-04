# End-to-End NLP Pipeline: From Regex to Dense Neural Embeddings 🚀

## 📌 Overview
To truly master Large Language Models (LLMs), one must understand the foundational mechanics that power them. This repository contains a comprehensive, end-to-end Natural Language Processing (NLP) system built from scratch. It bridges the gap between classical statistical methods and modern continuous vector spaces, showcasing a deep understanding of text processing architecture.

---

## 🛠️ Core Technical Milestones

### 1. Automated Data Acquisition & Linguistic Processing 🔍

**Web Scraping:**  
Built a robust pipeline to bypass dynamic DOM rendering using **Selenium** and **BeautifulSoup**.

**Information Extraction:**  
Developed advanced **Regex-driven tools** to parse and structure unstructured data.

**Linguistic Annotation:**  
Integrated **NLTK** for precise **Part-of-Speech (POS) tagging**, **lemmatization**, and **Named Entity Recognition (NER)**.

---

### 2. Statistical N-Gram Language Modeling 📊

**Generative Markov Models:**  
Implemented **Unigram** and **Bigram** language models entirely **from scratch**.

**Smoothing Techniques:**  
Applied **Laplace (Add-1) smoothing** and **Linear Interpolation** to gracefully handle **out-of-vocabulary (OOV)** context transitions.

**Evaluation:**  
Quantitatively evaluated models via **Perplexity** across both **in-domain** and **out-of-domain** text corpora.

---

### 3. Subword Tokenization (BPE) for Urdu 🌐

**Morphological Handling:**  
Tackled the complexity of **non-English languages lacking distinct word boundaries**.

**From-Scratch BPE:**  
Wrote a **Byte Pair Encoding (BPE)** algorithm from the ground up.

**Statistical Segmentation:**  
Iteratively merged frequent character pairs, enabling the model to successfully segment **unspaced Urdu text purely through statistical co-occurrence**.

---

### 4. Dense Neural Embeddings in PyTorch 🧠

**Sparse to Dense:**  
Transitioned from **classical sparse representations** to **continuous vectors**.

**Baselines:**  
Built **TF-IDF matrices** and **count-based Positive Pointwise Mutual Information (PPMI)** models.

**Neural Architecture:**  
Implemented a **Skip-Gram with Negative Sampling (SGNS)** neural network **natively in PyTorch**.

---

## 💡 Key Takeaways & Evaluation

The project concludes with both:

- **Intrinsic Evaluation:** Spearman correlation on **word similarity**
- **Extrinsic Evaluation:** **Topic Classification** using **Logistic Regression** on the **AG News dataset**

**Empirical Insight:**  
While **dense neural embeddings** represent the future of NLP, classical sparse representations like **TF-IDF** still pack a massive punch for **keyword-heavy classification tasks**, especially when **compute and training data are limited**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NLTK
- Selenium
- BeautifulSoup4
- Scikit-learn

---

### Installation

```bash
git clone https://github.com/yourusername/nlp-pipeline-from-scratch.git
cd nlp-pipeline-from-scratch
````
