# Evaluation Protocol & Benchmark Standards

This document establishes the evaluation protocol, metric definitions, and verification standards for evaluating answer sheet grading accuracy on the S3231 examination format.

---

## 1. Blind Evaluation Protocol

To prevent overfitting and maintain objective evaluation standards, the evaluation suite must be executed under strict blind testing rules:

1. **Reserved Test Suite:** All 90 images in `DATA_ROOT/stress_test/` (`p03`, `p04`, and `p19` across `scan`, `photo`, and `hard`) serve as an unseen evaluation benchmark.
2. **Zero Training Exposure:** Patches from `p03`, `p04`, and `p19` are excluded from all model training, validation loss calculation, and checkpoint selection.
3. **Frozen Hyperparameters:** All decision fusion thresholds, classical fill ratios, and alignment criteria must be tuned on the validation set (`vision_ds/labels/labels.csv` validation contributors) prior to running stress test evaluations. Thresholds must never be adjusted to suit individual test sheets.
4. **Read-Only Execution:** Stress test runs must never overwrite model weights or alter baseline ground truth files.

---

## 2. Standard Evaluation Metrics

### 2.1 Question-Level Answer Accuracy
Measures overall correctness of the graded output against ground truth:
$$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}$$
Where $N = 10,800$ total questions across 90 sheets ($120 \text{ questions} \times 90 \text{ sheets}$).

### 2.2 Multi-Mark Detection Metrics
Multi-mark questions (where two or more options are filled) are safety-critical. A valid OMR system must detect multiple marks and assign 0 marks rather than guessing one marked letter:
- **True Positive (TP):** Multiple marks present in ground truth and detected as `MULTIPLE_MARK`.
- **False Positive (FP):** Single mark or blank question incorrectly flagged as `MULTIPLE_MARK`.
- **False Negative (FN):** Multiple marks present in ground truth but recognized as a single option or blank.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 2.3 Mean Alignment Confidence
Telemetry score (0.0 to 1.0) derived from SIFT inlier ratio, affine condition number, and contour edge correlation. Scores $< 0.65$ trigger `LOW_ALIGNMENT_CONFIDENCE`.

---

## 3. Benchmark Ground Truth Performance (outputs/evaluation_report.json)

The official benchmark results recorded in `outputs/evaluation_report.json` across all 90 stress test sheets:

| Dimension | Slice | Sheets | Questions | Accuracy (%) | Invalid Prec | Invalid Rec | Invalid F1 | REVIEW rate | Mean Align Conf |
|---|---|---|---|---|---|---|---|---|---|
| **Condition** | `scan` | 30 | 3,600 | 93.92% | 75.56% | 48.57% | 59.13% | 1.78% | 0.712 |
| | `photo` | 30 | 3,600 | 93.86% | 75.00% | 47.14% | 57.89% | 2.83% | 0.697 |
| | `hard` | 30 | 3,600 | 92.06% | 68.89% | 44.29% | 53.91% | 2.64% | 0.699 |
| **Contributor** | `p03` | 30 | 3,600 | 92.97% | 100.00% | 35.00% | 51.85% | 3.92% | 0.754 |
| | `p04` | 30 | 3,600 | 98.42% | 92.16% | 65.28% | 76.42% | 0.39% | 0.714 |
| | `p19` | 30 | 3,600 | 88.44% | 48.39% | 38.46% | 42.86% | 2.94% | 0.641 |
| **Overall** | **All Sheets** | **90** | **10,800** | **93.28%** | **73.13%** | **46.67%** | **56.98%** | **2.42%** | **0.703** |

- Total questions evaluated: 10,800
- Total correct answers: 10,074 (93.28%)
- "Invalid" columns: ground truth `?` covers multiple marks and ticks, circles, crosses and faint marks (see Experiment 4 in docs/EXPERIMENT_LOG.md)
- Questions routed to REVIEW for a proctor: 261 (2.42%)
- Total execution time: 23.5 seconds ($\approx 0.26$ seconds per 120-question sheet, `--no-ocr`)

---

## 4. Evaluation History & Optimism Disclaimer

Over the course of development, four formal evaluation runs were conducted:
1. **Run 1 (Baseline MobileNetV3):** Initial model trained on single-person crops. Achieved 78.69% accuracy on stress test images, with multi-mark recall of 44.29%.
2. **Run 2 (Multi-Person Fine-Tuning & Decision Fusion):** Fine-tuned across 10 diverse training persons with CNN-confirmed decision fusion. Overall accuracy increased to 79.85%, multi-mark recall rose to 47.62%, and phone photo accuracy reached 83.53%.
3. **Run 3 (Metadata OCR Telemetry):** Added envelope padding and graceful fallback between TrOCR and EasyOCR for handwritten student metadata.
4. **Run 4 (Multi-Contributor Retraining & CNN-Veto Fusion):** Retrained across 42 contributors after fixing a patch-selection bug, and retuned fusion on validation contributors only. Overall accuracy 93.28%, invalid-answer precision 73.13%, 2.42% of questions routed to REVIEW. Rebuilt `stress_test/` from `vision_ds` images, so not strictly comparable with Runs 1 to 3.

### Optimism Disclaimer
The 93.28% accuracy benchmark is measured across 90 sheets from 3 contributors (`p03`, `p04`, `p19`). While these sheets represent challenging mobile photos, severe skew, and lighting gradients, real-world deployment across different institutions, varying paper weights, and alternative writing instruments (e.g., gel pens, markers) may exhibit variance. Systems deployed in high-stakes examination environments should always route flagged `REVIEW` and `LOW_ALIGNMENT_CONFIDENCE` sheets to human proctor verification.
