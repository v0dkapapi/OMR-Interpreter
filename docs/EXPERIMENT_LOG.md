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
