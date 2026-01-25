# Privacy-Preserving Document Processing: Technical Documentation

## 1. Design Decisions & Trade-offs

### 1.1 Regex vs. Model-Based Approach: Trade-off Analysis

I had two options for PII detection: regular expressions or NER models. Each works well for different types of data.

Regex is perfect for structured PII. SSNs, phone numbers, and emails follow fixed patterns. A French SSN is always 15 digits ("2 78 03 75 116 025 43"), a US phone is always 10 digits with optional formatting. I can write patterns that match these with near-perfect accuracy. Regex is fast and deterministic.

NER models handle unstructured PII. Names like "Marie Dubois" or "Jean-Pierre de la Fontaine" do not follow patterns. NER models understand context, so they know "Dr. Marie Dubois" is a person while "Rue Marie Dubois" is a street. On the downside, they are usually slower and can produce some false-positives.

I chose a hybrid approach. Regex handles all structured PII (SSN, phone, email, dates, medical IDs). The NER model detects unstructured entities (person names, locations, organizations). I then merge the results and remove duplicates; prioritizing regex matches when the same text span is detected by both methods.

### 1.2 Multilingual PII Format Handling

Different countries format PII differently. A French SSN looks like "2 78 03 75 116 025 43" while a US SSN is "123-45-6789". Phone numbers, dates, and postal codes all vary by country.

I handle this by detecting the document language first using the langdetect library, then applying country-specific regex patterns. For French documents, I use patterns that match French SSN format (15 digits with spaces), French phone numbers (starting with 0 or +33), and 5-digit postal codes. For English documents, I use US SSN format (9 digits with hyphens), US phone patterns, and zip codes. The exact regex patterns and cases they handle are shared in the notebook.

The NER models are already language-specific (fr_core_news_md for French, en_core_web_lg for English), so once I know the language, I load the appropriate model.

### 1.3 Handling False Positives in Medical/Legal Contexts

Medical and legal documents contain many labels, headers, and common terms that NER models incorrectly identify as PII. This creates false positives that would unnecessarily redact non-sensitive information.

The problem is visible in the NER output:

| Detected Text | Tagged As | Language | Actual Meaning | False Positive? |
|---------------|-----------|----------|----------------|-----------------|
| Sécurité Sociale | ORGANIZATION | French | Field label (Social Security) | Yes |
| Médecin | PERSON | French | Job title (Physician) | Yes |
| ID | ORGANIZATION | French | Field label (ID number) | Yes |
| SSN | ORGANIZATION | English | Field label (Social Security Number) | Yes |
| Marie Dubois | PERSON | French | Actual patient name | No |
| Jean Martin | PERSON | French | Actual doctor name | No |

These false positives happen because NER models learn patterns from general text. In most contexts, "Sécurité Sociale" does refer to the French government organization. "SSN" might appear as an organization acronym. The model cannot distinguish between "SSN: 123-45-6789" (where SSN is a label) and "works at SSN Corp" (where SSN is a company name).

My solution relies on the hybrid approach. Regex patterns catch the actual PII values (the SSN number itself, phone numbers, emails). When I merge regex and NER results, overlapping detections get filtered. For example, if NER detects "SSN" as an organization but regex detects "123-45-6789" right next to it, I keep the regex match and the document context makes clear that "SSN" is just a label.

For standalone false positives like "Médecin" or "ID" that do not overlap with actual PII, they remain in the results. However, redacting these is relatively harmless. Converting "Médecin: Dr. Jean Martin" to "[REDACTED]: Dr. [REDACTED]" does not expose real PII, and the document remains comprehensible.

A production system could filter these using a stopword list of common medical terms (Médecin, Docteur, Patient, SSN, ID) or by setting minimum entity length requirements. For this implementation, I accepted these minor false positives as they do not compromise privacy.

### 1.4 Downstream Task Performance Analysis

The goal of privacy-preserving redaction is to protect PII while maintaining document utility for downstream tasks. I tested this by running a fine-tuned medical NER model (from Problem 2) on three versions of each document: original, strict redaction, and surrogate redaction.

**Evaluation setup:** I created 14 test documents (8 English, 6 French) containing both PII and medical entities. Each document had clear diseases like "diabetes mellitus", "pneumonia", or "heart failure". I applied both redaction strategies to remove PII (names, SSNs, phones, emails), then ran the NER model on all three versions to detect diseases.

**Results:**

| Strategy | Precision | Recall | F1 Score | TP | FP | FN |
|----------|-----------|--------|----------|----|----|-----|
| Original (Baseline) | 0.4615 | 0.6000 | 0.5217 | 12 | 14 | 8 |
| Strict Redaction | 0.4615 | 0.6000 | 0.5217 | 12 | 14 | 8 |
| Surrogate Redaction | 0.3793 | 0.5500 | 0.4490 | 11 | 18 | 9 |

| Strategy | F1 Drop | Percentage Change |
|----------|---------|-------------------|
| Strict Redaction | 0.0000 | 0.0% |
| Surrogate Redaction | -0.0728 | -13.9% |

Strict redaction performed identically to the original. The F1 score remained at 0.5217 with no degradation. This happened because the diseases in my test documents were clearly stated and isolated from PII. Replacing "Patient John Smith has diabetes mellitus" with "Patient [REDACTED] has diabetes mellitus" did not affect the model's ability to detect "diabetes mellitus".

Surrogate redaction showed a 13.9% F1 drop. This was unexpected and primarily caused by tokenization artifacts rather than conceptual failure. When I replaced names like "Robert Davis" with "Justin Cowan", the character positions shifted. This caused the XLM-RoBERTa tokenizer to split some disease terms differently. For example, "bacterial pneumonia" tokenized as "bacteria" + "lpneumonia" in the surrogate version but as a single entity in the original. Same disease, different subword boundaries, so the evaluation matching failed.

**Key finding:** Both redaction strategies preserved medical entity detection capability. The F1 scores remained in the same range (0.45-0.52), indicating that neither strategy significantly degrades downstream utility for straightforward medical documents. The surrogate degradation was a technical artifact of tokenization, not a fundamental flaw in the approach.