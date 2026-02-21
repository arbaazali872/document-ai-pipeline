# Task B: Cross-Domain Document Classification (TF-IDF + Logistic Regression)

## Model selection rationale

I considered both transformer-based models (e.g., BERT) and classical machine learning approaches for this task, but selected TF-IDF vectorization with Logistic Regression due to data and task constraints. The dataset contained 180 documents in total (126 for training and 54 for testing), resulting in roughly 30–40 documents per class across Healthcare, Legal, Finance, and Multi-domain categories. Deep learning models generally require large amounts of data to generalize reliably, and with this limited dataset, a transformer-based approach would be prone to overfitting. In contrast, TF-IDF with Logistic Regression is better suited to small datasets and has been shown to perform consistently under such conditions.

The nature of the task also supported a simpler model, as the domains use largely distinct vocabularies. Healthcare documents commonly include terms such as “hypertension,” “patient,” and “diagnosis,” legal texts use “plaintiff,” “defendant,” and “jurisdiction,” while finance documents contain “revenue,” “equity,” and “securities.” This vocabulary difference means that keyword information alone is often sufficient to distinguish between classes, without requiring deeper semantic modeling. TF-IDF captures this by assigning higher weights to domain-related n-grams while reducing the influence of terms that appear across multiple document types.

From a practical perspective, the TF-IDF + Logistic Regression pipeline is computationally efficient, training in under a minute on CPU and requiring no GPU resources, which simplifies deployment. I treated this approach as a baseline, following the principle of starting with the simplest effective solution. The resulting accuracy of 98.15% suggests that the task is largely separable in TF-IDF feature space. Had performance fallen below 85–90%, a transformer-based model would have been considered, but the strong results indicate that additional model complexity would likely provide limited benefit given the dataset size.

# Hyperparameter Rationale

## Task A: Fine-tuning (XLM-RoBERTa)

**Learning Rate = 2e-5**  
Conservative choice within the BERT paper’s recommended range (2e-5 to 5e-5) to preserve multilingual representations and avoid catastrophic forgetting when fine-tuning on limited data (1,000 examples).  
*Source: Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers” (2019)*

**Epochs = 3**  
Validation F1 plateaued after epoch 2, with only marginal improvement at epoch 3. Additional epochs increased the risk of overfitting on the small dataset (500 EN + 500 FR).

**Batch Size = 16**  
Limited by GPU memory (T4). Medical abstracts often approach the 512-token limit, and larger batch sizes caused out-of-memory errors.

**Weight Decay = 0.01**  
Standard L2 regularization from the BERT paper to reduce overfitting to domain-specific medical terminology.  
*Source: Devlin et al., “BERT” (2019)*

**Warmup Steps = 100**  
Covers approximately 54% of the first epoch, helping stabilize the randomly initialized classification head before fully updating the pre-trained encoder.

**Max Length = 512**  
Maximum supported by XLM-RoBERTa; necessary as medical abstracts frequently reach this length.

---

## Task B: TF-IDF + Logistic Regression

**max_features = 3000**  
Balances coverage of domain-specific vocabulary (medical, legal, financial) with computational efficiency. Experiments showed that increasing beyond 5,000 features yielded less than 0.5% accuracy improvement.

**ngram_range = (1, 2)**  
Unigrams and bigrams capture key multi-word expressions such as legal phrases (“European Court,” “Human Rights”). Trigrams increased the feature space by ~30× with minimal benefit on the 180-document dataset.


# Performance Analysis with Visualizations

## Task A: Multilingual NER Results

I evaluated the model across English and French test sets to understand how well the multilingual training transferred across languages, and the results revealed a significant performance gap that I attribute to differences in how many entities each dataset actually contains.

| Configuration | Language | F1 Score | Precision | Recall | Training Data |
|--------------|----------|----------|-----------|--------|---------------|
| Bilingual Model | English | 0.7672 | 0.7339 | 0.8036 | 500 EN + 500 FR |
| Bilingual Model | French | 0.4762 | 0.5191 | 0.4398 | 500 EN + 500 FR |
| Zero-Shot Transfer | French | 0.4096 | 0.3868 | 0.4352 | 500 EN only |

![Performance Comparison](./data/download.png)

The English test set reached an F1 of 0.7672 with a high recall of 0.8036, showing that the model learned to identify disease entities well despite the small training set of 500 documents. However, the lower precision of 0.7339 indicates some over-prediction, where non-disease terms are occasionally labeled as diseases. French performance was much weaker, with an F1 of 0.4762, likely because the NCBI dataset contained 5,130 disease entities across 500 documents, while Quaero included only 888 entities in the same number of documents, giving the model far fewer French examples during training.

In the zero-shot setting, the model achieved an F1 of 0.4096 on French when trained only on English data, just 0.0666 points below the bilingual model’s French score. This shows that XLM-RoBERTa’s cross-lingual representations enable strong transfer even without target-language data. The limited gain from adding 500 French documents suggests that 888 entities were insufficient to provide meaningful improvement beyond what cross-lingual transfer already offered.

---
## Task B: Cross-Domain Classification Results

I evaluated the TF-IDF + Logistic Regression classifier on the test set of 54 documents to measure how accurately it distinguishes between Healthcare, Legal, Finance, and Multi-domain categories, and the results showed near-perfect classification performance with only a single misclassification across all test cases. I used a 70/30 train-test split on the total dataset of 180 documents, resulting in 126 training documents and 54 test documents, where the test set contained 15 Healthcare documents, 15 Legal documents, 15 Finance documents, and 9 Multi-domain documents, maintaining roughly the same class distribution as the full dataset which had 50 documents each for the three single-domain categories and 30 multi-domain documents.

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Finance | 1.0000 | 1.0000 | 1.0000 | 15 |
| Healthcare | 0.9375 | 1.0000 | 0.9677 | 15 |
| Legal | 1.0000 | 1.0000 | 1.0000 | 15 |
| Multi-domain | 1.0000 | 0.8889 | 0.9412 | 9 |
| **Overall Accuracy** | - | - | - | **0.9815** |

![Confusion Matrix](./data/Task_B_2.png)

The overall accuracy of 98.15% (53 out of 54 test documents) shows that the domains have clearly distinct vocabularies, allowing simple keyword-based classification to perform very well. Legal documents use terms like “plaintiff,” “defendant,” and “jurisdiction,” finance documents include “securities,” “revenue,” and “equity,” and healthcare documents contain terms such as “diagnosis,” “patient,” and disease names. Legal and Finance achieved perfect scores across all metrics, reflecting the highly specialized language in the MAPA and Gretel datasets.

Healthcare achieved an F1 of 0.9677 with perfect recall (1.0000) but slightly lower precision (0.9375), meaning one non-healthcare document was misclassified as healthcare. This document was multi-domain. The Multi-domain class had the lowest F1 of 0.9412, with perfect precision but lower recall (0.8889), likely due to the synthetic construction of these texts, where one domain’s vocabulary can dominate the TF-IDF features and lead to misclassification.

---
# Error analysis
## Error Analysis with Concrete Examples

### Example EN-1 (English)

**Word-level truth:**  
ataxia-telangiectasia

**Subword prediction:**  
▁ata(1) xia(1) -(2) tel(2) angi(2) ecta(2) sia(2)

**Analysis:**  
I observe that the model correctly detects the disease term, but it splits the word into multiple subwords. Because of this, the model assigns more than one beginning label to the same entity. This happens due to tokenization and not because the model misunderstands the medical term. The full entity is still captured, but its boundaries are not perfectly aligned.

---

### Example EN-2 (English)

**Word-level truth:**  
Autosomal dominant neurohypophyseal diabetes insipidus

**Subword prediction:**  
▁Auto(1) som(1) al(2) ▁dominant(2) ▁neuro(2) hy(2) po(2) phy(2) se(2) al(2) ▁diabetes(2) ▁in(2) si(2) pid(2) us(2)

**Analysis:**  
In this case, the model successfully identifies the entire disease name despite heavy subword splitting. The begin and inside labels remain consistent across many tokens. This shows that the model can maintain entity coherence even for long and complex medical terms.

---

### Example FR-1 (French)

**Word-level truth:**  
lipome retroperitoneal

**Subword prediction:**  
▁lipo(1) me(2) ▁retro(2) peri(2) tone(2) al(2)

**Analysis:**  
The model correctly detects both French medical terms and assigns appropriate labels across subwords. Although the words are split into several pieces, the entity spans remain intact. This shows that the model handles common French medical compounds well when token boundaries are regular.

---

## Concrete Examples (Task B)

### Example 1 – Correct Classification

**Predicted:** Finance | **Confidence:** 0.714  
**True Label:** Finance  

**Text:**  
“Subject: Important Notice: Your Coverage Extension Renewal is Approaching. Dear Policyholder, we are writing to inform you that your coverage extension renewal is approaching… **Renewal Date:** 01/01/2023… **Premium Amount:** $[Premium Amount]… Thank you for choosing us for your insurance needs.”

---

### Example 2 – Correct Classification

**Predicted:** Multi-domain | **Confidence:** 0.540  
**True Label:** Multi-domain  

**Text:**  
“Hereditary deficiency of the fifth component of complement in man… Clinical, immunochemical, and family studies. **PROCEDURE** The case originated in applications against the Republic of Turkey lodged with the European Commission of Human Rights under Article 25 of the Convention…”

---

### Example 3 – Misclassification

**Predicted:** Healthcare | **Confidence:** 0.399  
**True Label:** Multi-domain  

**Text:**  
“Low frequency of BRCA1 germline mutations in German breast/ovarian cancer families… **FINANCIAL DISCLOSURE STATEMENT**… Government grants and subsidies tracker… Grant Amount: £2,500,000…”

---

### Discussion

These examples shows both the strengths and limitations of the cross-domain classifier. Single-domain documents with clearly defined vocabulary, such as finance-related communications, are classified correctly with high confidence. Documents containing mixed docs(health+finance, finance+legal, legal+health) from multiple domains can also be successfully identified as multi-domain. However, the misclassified example shows that when one domain’s terminology dominates the text, secondary domain signals may be underweighted, leading to a multi-domain document being incorrectly labeled as single-domain.

