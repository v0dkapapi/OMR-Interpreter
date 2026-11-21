"""Diagnostic script to verify bubble patch intensity and alignment of template ROIs against a canonical sample sheet."""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

with open(config.TEMPLATE_PATH) as f:
    template = json.load(f)

img = cv2.imread(str(config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg"))
canonical = cv2.resize(img, (config.CANONICAL_WIDTH, config.CANONICAL_HEIGHT))
gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)

print("Checking intensity of template ROIs on p01-f01 (Ground truth: Q1=E, Q2=E, Q3=C, Q4=E, Q5=C):")
for q in range(1, 6):
    q_key = str(q)
    rois = template["bubble_rois"][q_key]
    intensities = {}
    for opt in ["A", "B", "C", "D", "E"]:
        roi = rois[opt]
        patch = gray[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
        intensities[opt] = float(np.mean(patch))
    darkest_opt = min(intensities, key=intensities.get)
    print(f"  Q{q}: Intensities = { {k: round(v, 1) for k, v in intensities.items()} } -> Darkest: {darkest_opt}")
