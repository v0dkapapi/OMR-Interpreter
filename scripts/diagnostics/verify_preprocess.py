"""Diagnostic script to visually verify preprocessing and alignment across scan, photo, and hard capture sheets."""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.preprocess import SheetPreprocessor

def overlay_template_rois(img_bgr: np.ndarray, template: dict) -> np.ndarray:
    """Overlay template ROIs on canonical image for visual inspection."""
    canvas = img_bgr.copy()
    
    # 1. Anchors
    anchors = template.get("alignment_anchors", {})
    for name, anchor in anchors.items():
        x, y, w, h = anchor["x"], anchor["y"], anchor["w"], anchor["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(canvas, name, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 2. Handwritten
    handwritten = template.get("handwritten_rois", {})
    for name, roi in handwritten.items():
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(canvas, name, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # 3. Student Number bubbles
    sn_grid = template.get("student_number_grid", {})
    for col_bubbles in sn_grid.values():
        for digit, roi in col_bubbles.items():
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2)
            cv2.circle(canvas, (cx, cy), r, (0, 200, 0), 1)

    # 4. Question bubbles
    q_rois = template.get("bubble_rois", {})
    for q_num, opts in q_rois.items():
        for opt, roi in opts.items():
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2)
            cv2.circle(canvas, (cx, cy), r, (0, 0, 200), 1)

    return canvas

def main():
    preprocessor = SheetPreprocessor()
    with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    # Select representative test cases from p01 (NOT in p03, p04, p19)
    test_cases = [
        ("scan", config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg"),
        ("photo", config.VISION_IMAGES_DIR / "p01" / "p01-f01-photo.jpg"),
        ("hard", config.VISION_IMAGES_DIR / "p01" / "p01-f01-hard.jpg")
    ]

    # Verify none of the test cases involve excluded persons
    for kind, path in test_cases:
        for excl in config.EXCLUDED_PERSONS:
            assert excl not in str(path), f"Violation: {excl} found in test path {path}!"

    print("=== STARTING PREPROCESSING VISUAL VERIFICATION ===")
    results = {}
    thumbnails = []

    for kind, img_path in test_cases:
        print(f"\nProcessing {kind.upper()} ({img_path.name})...")
        res = preprocessor.process(img_path)
        results[kind] = res

        print(f"  Method: {res.method_used}")
        print(f"  Confidence Score: {res.confidence:.4f}")
        print(f"  Flags: {res.flags}")
        print(f"  Metadata: {res.metadata}")

        # Create overlay visualization
        overlay = overlay_template_rois(res.canonical_bgr, template)

        # Draw HUD banner with confidence and method
        hud_text = f"Condition: {kind.upper()} | Method: {res.method_used} | Confidence: {res.confidence:.2f} | Flags: {res.flags or 'NONE'}"
        color = (0, 180, 0) if res.is_reliable else (0, 0, 220)
        cv2.rectangle(overlay, (0, 0), (config.CANONICAL_WIDTH, 45), (255, 255, 255), -1)
        cv2.rectangle(overlay, (0, 0), (config.CANONICAL_WIDTH, 45), color, 2)
        cv2.putText(overlay, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (17, 24, 39), 2)

        # Save individual verification outputs
        out_path = config.OUTPUTS_DIR / f"preprocess_{kind}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"  Saved visual overlay to: {out_path}")

        # Save normalized thresholded preview
        thresh_path = config.OUTPUTS_DIR / f"preprocess_{kind}_thresh.jpg"
        cv2.imwrite(str(thresh_path), res.thresholded)

        # Create thumbnail for composite
        thumb = cv2.resize(overlay, (500, int(500 * config.CANONICAL_HEIGHT / config.CANONICAL_WIDTH)))
        thumbnails.append(thumb)

    # Save side-by-side composite comparison
    composite = np.hstack(thumbnails)
    comp_path = config.OUTPUTS_DIR / "preprocess_comparison.jpg"
    cv2.imwrite(str(comp_path), composite)
    print(f"\nSaved composite 3-condition comparison to: {comp_path}")

    # Check question alignment accuracy on first 10 questions against ground truth
    # In p01-f01, ground truth Q1..Q10 are: E, E, C, E, C, E, D, A, A, B
    ground_truth = {1: "E", 2: "E", 3: "C", 4: "E", 5: "C", 6: "E", 7: "D", 8: "A", 9: "A", 10: "B"}
    print("\n=== BUBBLE ALIGNMENT SANITY CHECK (Q1..Q10) ===")
    for kind, res in results.items():
        norm_img = res.normalized_gray
        matches = 0
        predictions = {}
        for q in range(1, 11):
            q_key = str(q)
            rois = template["bubble_rois"][q_key]
            intensities = {}
            for opt, roi in rois.items():
                patch = norm_img[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
                intensities[opt] = float(np.mean(patch))
            darkest = min(intensities, key=intensities.get)
            predictions[q] = darkest
            if darkest == ground_truth[q]:
                matches += 1

        print(f"  Condition: {kind:<6} | Q1-Q10 Darkest vs Ground Truth: {matches}/10 correct | Predictions: {list(predictions.values())}")

if __name__ == "__main__":
    main()
