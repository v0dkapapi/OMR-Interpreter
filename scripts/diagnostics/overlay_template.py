"""Diagnostic script to overlay template.json ROIs onto an aligned canonical sheet for visual inspection."""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

def draw_roi(img, roi, label, color=(0, 0, 255), thickness=2):
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["w"])
    h = int(roi["h"])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def overlay(image_path: Path, template_path: Path, output_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize to canonical dimensions if necessary
    target_w = template.get("page_size", {}).get("width", config.CANONICAL_WIDTH)
    target_h = template.get("page_size", {}).get("height", config.CANONICAL_HEIGHT)
    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    overlay_img = img.copy()

    # 1. Draw alignment anchors
    anchors = template.get("alignment_anchors", {})
    for name, anchor in anchors.items():
        x, y, w, h = anchor["x"], anchor["y"], anchor["w"], anchor["h"]
        cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (255, 0, 0), 3) # Blue for anchors
        cv2.putText(overlay_img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 2. Draw handwritten ROIs
    handwritten = template.get("handwritten_rois", {})
    for field_name, roi in handwritten.items():
        draw_roi(overlay_img, roi, field_name, color=(0, 165, 255), thickness=2) # Orange for text

    # 3. Draw Student ID grid ROIs
    student_id_grid = template.get("student_number_grid", {})
    for col_idx, col_bubbles in student_id_grid.items():
        for digit, roi in col_bubbles.items():
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2)
            cv2.circle(overlay_img, (cx, cy), r, (0, 200, 0), 1) # Green for ID bubbles

    # 4. Draw Question Bubble ROIs
    questions = template.get("bubble_rois", {})
    for q_num, options in questions.items():
        for opt, roi in options.items():
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2)
            # Red/maroon outline for bubbles
            cv2.circle(overlay_img, (cx, cy), r, (0, 0, 180), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_img)
    print(f"Overlay successfully written to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay template ROIs on an answer sheet.")
    parser.add_argument("--image", type=Path, default=config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg", help="Path to sheet image")
    parser.add_argument("--template", type=Path, default=config.TEMPLATE_PATH, help="Path to template.json")
    parser.add_argument("--output", type=Path, default=config.OUTPUTS_DIR / "template_overlay.jpg", help="Output path")
    args = parser.parse_args()

    overlay(args.image, args.template, args.output)
