# Machine Learning & Pipeline Experiment Log

This document records the experimental milestones, model iterations, validation metrics, and benchmark results obtained during development.

---

## Experiment 1: Baseline Classical vs Initial MobileNetV3

- **Date:** November 17, 2026
- **Architecture:** Classical Otsu circular fill ratio vs initial MobileNetV3-Small classifier.
- **Training Data:** Single-person subset crops from `p01` and `p02`.
- **Validation Results:**
  - Classical fill-ratio baseline accuracy on validation set: 55.46%
  - Initial MobileNetV3 accuracy: 64.20%
- **Stress Test Evaluation (90 sheets):**
  - Overall accuracy: 78.69% (8,498 / 10,800 questions)
  - Multi-mark recall: 44.29% (93 / 210 detected)
  - Key finding: High rate of disagreement between classical fill ratio and CNN on faint pencil marks.

---

## Experiment 2: Multi-Person Fine-Tuning & CNN-Confirmed Decision Fusion

- **Date:** November 18, 2026
- **Objective:** Improve multi-mark recall and phone photo resilience without overfitting to specific contributors.
- **Dataset Splitting:**
  - Training: 7,492 balanced bubble patches extracted across 10 training persons (`p01`, `p02`, `p05`, `p06`, `p07`, `p08`, `p09`, `p11`, `p12`, `p13`).
  - Validation: 2,454 patches across 5 disjoint validation persons (`p14`, `p15`, `p16`, `p17`, `p18`).
  - Leakage Exclusion: `p03`, `p04`, and `p19` strictly excluded.
- **Model Training:**
  - Architecture: MobileNetV3-Small (PyTorch MPS backend, 4 epochs, learning rate 1e-3, AdamW optimizer, CrossEntropyLoss).
  - Validation Accuracy: 67.11% (1,647 / 2,454 individual bubble crops).
  - Classical Baseline on same val set: 55.46% (1,361 / 2,454).
  - Model vs Baseline Disagreements: 1,310 (53.38%).
- **Decision Fusion Rules:**
  - If CNN predicts `filled` with probability $> 0.50$ and classical fill ratio $> 0.38$, confirm filled.
  - Multi-mark condition: detect when multiple options exceed fill threshold and are confirmed by classifier.
- **Stress Test Evaluation (Official Benchmark):**
  - **Overall Accuracy:** 79.85% (8,624 / 10,800 questions)
  - **Multi-Mark Recall:** 47.62% (100 / 210 detected, up from 44.29%)
  - **Multi-Mark Precision:** 34.01%
  - **Multi-Mark F1:** 39.68%
  - **Condition Breakdown:**
    - `scan` (30 sheets): 79.28% accuracy, 47.14% multi-mark recall
    - `photo` (30 sheets): 83.53% accuracy, 50.00% multi-mark recall
    - `hard` (30 sheets): 76.75% accuracy, 45.71% multi-mark recall
  - **Contributor Breakdown:**
    - `p03` (sloppy/pencil): 78.47% accuracy, 53.33% recall
    - `p04` (phone photos/skew): 82.36% accuracy, 61.11% recall
    - `p19` (lighting gradients): 78.72% accuracy, 30.77% recall

---

## Experiment 3: Metadata OCR Telemetry & Dual-Engine Integration

- **Date:** November 19-20, 2026
- **Objective:** Extract student metadata (Name, Student ID, Exam Code, Seat Number, Date) from aligned sheet ROIs.
- **Architecture:** Dual-engine OCR:
  - Primary: TrOCR (`microsoft/trocr-base-handwritten`)
  - Offline Fallback: EasyOCR (CPU)
  - 8-digit bubble grid reader for Student ID cross-checking.
- **Enhancements Implemented:**
  - ROI bounding box margin expansion to prevent clipping ascenders/descenders on loose handwriting.
  - Alphanumeric code post-processing for module codes (`EEN32930`) and seat numbers (`F-376`).
  - Structured error fallback when local neural engine runs in restricted sandbox or CPU environments.

---

## Experiment 4: Multi-Contributor Retraining & CNN-Veto Decision Fusion

- **Date:** November 22, 2026
- **Dataset:** Fresh download of S3231 v1.0 (2,184 images, 740 forms). `stress_test/` rebuilt with `scripts/setup_dataset.py` as symlinks to the `p03`, `p04`, `p19` images in `vision_ds/images/`. The original `stress_test/` captures used for Experiments 1 to 3 had different file names (for example `p03-01-scan.jpg`, `hardphoto`, `.jpeg`), so earlier numbers are not strictly comparable.
- **Bug found in patch extraction:** `build_cache(max_forms=N)` took the first N rows of `labels.csv`, which is sorted by contributor, so `--max-forms 15` trained on a single person (`p07`) and validated on a single person (`p01`). Fixed with round-robin selection across contributors (`_round_robin_by_contributor`).
- **Model Training:** MobileNetV3-Small (ImageNet-pretrained), 5 epochs, `--max-forms 90 --subset 20000`.
  - Training: 14,400 patches from 42 contributors. Validation: 5,600 patches from 29 disjoint contributors. `p03`, `p04`, `p19` excluded.
  - Validation patch accuracy: 92.84% (93.27% best epoch).
- **Error analysis with the previous fusion rules (stress test, 75.20%):** 58% of errors were blank questions read as answered, 22% marked answers read as blank, 9% single answers called multi-mark. The fill-margin rule accepted the darkest bubble without consulting the CNN, so printed letters and small misalignment on unused rows became answers.
- **Fusion retuning (validation contributors only, 175 sheets from 29 people, forms not used during training):**

  | Variant | Validation accuracy | Invalid precision | REVIEW rate |
  |---|---|---|---|
  | Previous rules | 73.58% | 17% | 0.0% |
  | + CNN veto on fill-margin answers | 90.62% | 17% | n/a |
  | + CNN acceptance of faint marks | 92.50% | 17% | n/a |
  | + CNN-confidence multi-mark rule (conf > 0.8) | 93.44% | 89% | 14.4% (too broad) |
  | + narrowed disagreement rules (final) | **93.56%** | **89%** | **1.83%** |

- **Stress test (evaluated once, after tuning was frozen):**

| Dimension | Slice | Sheets | Questions | Accuracy (%) | Invalid Prec | Invalid Rec | Invalid F1 | REVIEW rate | Mean Align Conf |
|---|---|---|---|---|---|---|---|---|---|
| **Condition** | `scan` | 30 | 3,600 | 93.92% | 75.56% | 48.57% | 59.13% | 1.78% | 0.712 |
| | `photo` | 30 | 3,600 | 93.86% | 75.00% | 47.14% | 57.89% | 2.83% | 0.697 |
| | `hard` | 30 | 3,600 | 92.06% | 68.89% | 44.29% | 53.91% | 2.64% | 0.699 |
| **Contributor** | `p03` | 30 | 3,600 | 92.97% | 100.00% | 35.00% | 51.85% | 3.92% | 0.754 |
| | `p04` | 30 | 3,600 | 98.42% | 92.16% | 65.28% | 76.42% | 0.39% | 0.714 |
| | `p19` | 30 | 3,600 | 88.44% | 48.39% | 38.46% | 42.86% | 2.94% | 0.641 |
| **Overall** | **All Sheets** | **90** | **10,800** | **93.28%** | **73.13%** | **46.67%** | **56.98%** | **2.42%** | **0.703** |

- **Fill-margin-only mode (no weights):** 71.43% on the stress test, with no multi-mark detection. Using the fill-ratio class as a stand-in CNN dropped this to 36.17%, so that fallback was removed.
- **Label semantics note:** In this dataset version every invalid answer is labelled `?` (43 true multiple marks plus 27 circles, ticks, crosses and faint marks on the stress forms); no `BD`-style labels exist. The "multi-mark" metrics therefore measure invalid-answer detection.

