# Multilingual Medical NER & Domain-Aware Document Classification

A two-part NLP pipeline for clinical and multi-domain document understanding. The first part fine-tunes a multilingual transformer to detect disease mentions in English and French medical text. The second part builds a document classification and entity routing system that identifies whether a document belongs to Healthcare, Legal, Finance, or multiple domains — then runs the appropriate NER extractor for each.

---

## Overview

### Part 1 — Fine-tuned Multilingual NER

Fine-tunes `xlm-roberta-base` on two real medical corpora to detect disease entities in both English and French using a unified BIO label schema (`B-Disease`, `I-Disease`, `O`). Also includes a zero-shot cross-lingual transfer experiment to quantify how much XLM-RoBERTa's built-in multilingual representations contribute versus supervised French training data.

**Datasets:**
- [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) — English biomedical abstracts with disease annotations (5,130 entities across 500 training documents)
- [Quaero FrenchMed](https://quaerofrenchmed.limsi.fr/) — French medical text in BRAT format, DISO entities only (888 entities across 500 training documents)

**Model:** `xlm-roberta-base` — chosen over mBERT for stronger cross-lingual transfer, especially on low-resource language pairs.

**Training setup:**
- 500 documents sampled from each language, combined and shuffled
- Subword tokenization with proper label alignment (only the first subword of each word gets the real label)
- 3 epochs, lr=2e-5, batch size=16, mixed precision on GPU
- Evaluated separately on English and French test sets

**Results:**

| Configuration | Language | F1 | Precision | Recall |
|---|---|---|---|---|
| Bilingual (EN + FR) | English | 0.7672 | 0.7339 | 0.8036 |
| Bilingual (EN + FR) | French | 0.4762 | 0.5191 | 0.4398 |
| Zero-Shot (EN only) | French | 0.4096 | 0.3868 | 0.4352 |

The zero-shot model sits only 0.067 F1 points below the bilingual model on French, showing that XLM-RoBERTa's cross-lingual representations transfer well even without any French supervision. The limited gain from adding 500 French documents is largely attributed to the lower entity density in Quaero (888 vs 5,130 entities for the same number of documents).

---

### Part 2 — Domain Classification & Entity Routing

Classifies documents into Healthcare, Legal, Finance, or Multi-domain using TF-IDF + Logistic Regression, then routes each document to the appropriate NER extractor based on the predicted class.

**Why TF-IDF + Logistic Regression and not a transformer?**
The dataset contains 180 documents (126 train / 54 test), roughly 30–40 per class. At this scale, a transformer would overfit. The three domains use highly distinct vocabularies — medical terms, legal terminology, financial language — so keyword-level features are sufficient. The model trained in under a minute on CPU and achieved 98.15% accuracy, which left no practical reason to add model complexity.

**Datasets:**
- Healthcare: NCBI Disease test set (reused from Part 1)
- Legal: [MAPA Text Anonymization Benchmark](https://huggingface.co/datasets/mattmdjaga/text-anonymization-benchmark-train)
- Finance: [Gretel Synthetic PII Finance](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) (English only)
- Multi-domain: 30 synthetic documents constructed by splicing sentences from two different domain documents

**Entity routing:**
- Healthcare → fine-tuned XLM-RoBERTa from Part 1 (disease entities)
- Legal → spaCy `en_core_web_trf` filtered to PERSON, ORG, DATE, GPE, LAW, EVENT
- Finance → spaCy `en_core_web_trf` filtered to MONEY, ORG, PERCENT, CARDINAL, DATE
- Multi-domain → all three extractors run in parallel

**Classification results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Finance | 1.0000 | 1.0000 | 1.0000 |
| Healthcare | 0.9375 | 1.0000 | 0.9677 |
| Legal | 1.0000 | 1.0000 | 1.0000 |
| Multi-domain | 1.0000 | 0.8889 | 0.9412 |
| **Overall** | | | **0.9815** |

The single misclassification was a multi-domain document with dominant healthcare vocabulary, which was predicted as Healthcare. This is a known limitation of TF-IDF when one domain's signal overwhelms the other in mixed documents.


---

## Setup

The notebook is designed to run on Google Colab with a T4 GPU. Mount your Google Drive to load/save the fine-tuned model.

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 2e-5 | Conservative to avoid catastrophic forgetting on limited data |
| Epochs | 3 | F1 plateaued after epoch 2 |
| Batch size | 16 | T4 GPU memory constraint with 512-token sequences |
| Weight decay | 0.01 | Standard L2 regularization (Devlin et al., 2019) |
| Warmup steps | 100 | Stabilizes randomly initialized classification head |
| Max length | 512 | Medical abstracts frequently hit this limit |