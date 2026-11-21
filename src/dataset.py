"""Dataset loader and bubble patch extractor for OMR answer sheets.

Features:
- Verifies physical file existence on disk (handles p10 CSV/disk discrepancy)
- Strictly excludes p03, p04, and p19 from train/val splits
- Maps ground truth answers, erasures, and multi-marks to unfilled, filled, ambiguous
- Crops and caches 48x48 patches under cache/patches/ (git-ignored)
- PyTorch BubbleDataset with augmentation support and person-based splitting
"""

import os
import csv
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict, Any, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

import config
from src.preprocess import SheetPreprocessor

logger = logging.getLogger(__name__)


def parse_erased_and_sloppy(changed_str: str, sloppy_str: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Parse changed (e.g. 'q11:E>C; q39:C>B') and sloppy (e.g. 'q12:circle B; q21:multiple CD')."""
    erased = {} # q_num -> erased option(s)
    sloppy = {} # q_num -> sloppy description

    if changed_str and changed_str.strip():
        for item in changed_str.split(";"):
            item = item.strip()
            if ":" in item and ">" in item:
                q_part, change_part = item.split(":", 1)
                q_clean = "".join(filter(str.isdigit, q_part))
                if q_clean:
                    old_opt, new_opt = change_part.split(">", 1)
                    erased[int(q_clean)] = old_opt.strip()

    if sloppy_str and sloppy_str.strip():
        for item in sloppy_str.split(";"):
            item = item.strip()
            if ":" in item:
                q_part, desc_part = item.split(":", 1)
                q_clean = "".join(filter(str.isdigit, q_part))
                if q_clean:
                    sloppy[int(q_clean)] = desc_part.strip()

    return erased, sloppy


class PatchExtractor:
    """Extracts and caches bubble patches from preprocessed canonical sheets."""

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        patch_size: Tuple[int, int] = config.BUBBLE_PATCH_SIZE,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.template_path = Path(template_path) if template_path else config.TEMPLATE_PATH
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir) if cache_dir else config.PATCH_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.csv"

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = json.load(f)

        self.bubble_rois = self.template.get("bubble_rois", {})
        self.student_grid = self.template.get("student_number_grid", {})
        self.preprocessor = SheetPreprocessor(template_path=self.template_path)

    def extract_from_form(
        self,
        image_path: Path,
        label_row: Dict[str, str],
        person_split: str
    ) -> List[Dict[str, Any]]:
        """Extract all 120x5 question bubbles and student ID bubbles from a form image."""
        if not image_path.exists():
            return []

        form_id = label_row["form_id"]
        contributor = label_row["contributor"]
        condition = image_path.stem.split("-")[-1]  # scan, photo, hard

        # Preprocess sheet to canonical frame
        try:
            res = self.preprocessor.process(image_path)
        except Exception as e:
            logger.warning(f"Failed to preprocess {image_path}: {e}")
            return []

        norm_gray = res.normalized_gray
        erased_map, sloppy_map = parse_erased_and_sloppy(
            label_row.get("changed", ""), label_row.get("sloppy", "")
        )

        patches_metadata = []

        # 1. Extract Question Bubbles (Q1 to Q120, Options A-E)
        for q_idx in range(1, 121):
            q_key = str(q_idx)
            if q_key not in self.bubble_rois:
                continue

            q_col = f"q{q_idx}"
            gt_ans = label_row.get(q_col, "-").strip()
            erased_opts = erased_map.get(q_idx, "")
            is_sloppy = q_idx in sloppy_map

            for opt in ["A", "B", "C", "D", "E"]:
                roi = self.bubble_rois[q_key][opt]
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

                # Add a 6-pixel contextual margin around the bubble ROI
                m = 6
                y1 = max(0, y - m)
                y2 = min(norm_gray.shape[0], y + h + m)
                x1 = max(0, x - m)
                x2 = min(norm_gray.shape[1], x + w + m)

                patch = norm_gray[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                # Resize to target CNN input dimension (48x48)
                patch_resized = cv2.resize(patch, self.patch_size, interpolation=cv2.INTER_AREA)

                # Determine patch ground truth class
                if opt in erased_opts:
                    class_name = "ambiguous"
                elif is_sloppy and opt in sloppy_map[q_idx]:
                    class_name = "ambiguous"
                elif gt_ans == "?":
                    class_name = "ambiguous"
                elif opt in gt_ans:
                    class_name = "filled"
                else:
                    class_name = "unfilled"

                class_idx = config.CLASS_TO_IDX[class_name]

                # Save patch image to cache
                patch_filename = f"{form_id}_{condition}_q{q_idx}_{opt}.png"
                patch_path = self.cache_dir / patch_filename
                cv2.imwrite(str(patch_path), patch_resized)

                patches_metadata.append({
                    "patch_path": str(patch_path.relative_to(config.PROJECT_ROOT)),
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "form_id": form_id,
                    "person": contributor,
                    "split": person_split,
                    "condition": condition,
                    "type": "question",
                    "target": f"q{q_idx}_{opt}",
                })

        return patches_metadata

    def build_cache(
        self,
        max_forms: Optional[int] = None,
        force_rebuild: bool = False
    ) -> List[Dict[str, Any]]:
        """Scan vision_ds/labels/labels.csv, extract patches, and create manifest.csv."""
        if self.manifest_path.exists() and not force_rebuild:
            logger.info(f"Loading existing patch manifest from {self.manifest_path}")
            records = []
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["class_idx"] = int(r["class_idx"])
                    records.append(r)
            if records:
                return records

        labels_path = config.VISION_LABELS_DIR / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found at {labels_path}")

        records = []
        forms_processed = 0

        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            label_rows = list(reader)

        logger.info(f"Scanning {len(label_rows)} forms from labels.csv...")
        
        # Group available forms by split to ensure balanced cache
        train_rows = [r for r in label_rows if r["contributor"] not in config.EXCLUDED_PERSONS and r["split"] == "train"]
        val_rows = [r for r in label_rows if r["contributor"] not in config.EXCLUDED_PERSONS and r["split"] in ("val", "test")]

        selected_rows = []
        if max_forms:
            n_train = max(1, max_forms * 2 // 3)
            n_val = max(1, max_forms - n_train)
            selected_rows = train_rows[:n_train] + val_rows[:n_val]
        else:
            selected_rows = train_rows + val_rows

        for row in selected_rows:
            person = row["contributor"]
            form_id = row["form_id"]
            person_split = row["split"]
            person_dir = config.VISION_IMAGES_DIR / person

            if not person_dir.exists():
                continue

            for kind in ["scan", "photo", "hard"]:
                img_path = person_dir / f"{form_id}-{kind}.jpg"
                if not img_path.exists():
                    alt = person_dir / f"{form_id}-{kind}.jpeg"
                    if alt.exists():
                        img_path = alt
                    else:
                        continue

                patches = self.extract_from_form(img_path, row, person_split)
                records.extend(patches)

            forms_processed += 1

        # Save manifest
        if records:
            fieldnames = [
                "patch_path", "class_name", "class_idx", "form_id",
                "person", "split", "condition", "type", "target"
            ]
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        logger.info(f"Cached {len(records)} patches across {forms_processed} forms to {self.manifest_path}")
        return records


class BubbleDataset(Dataset):
    """PyTorch Dataset for cropped bubble patches with data augmentations."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        is_train: bool = True,
        transform: Optional[Any] = None
    ):
        self.records = records
        self.is_train = is_train

        if transform is not None:
            self.transform = transform
        elif is_train:
            # Training augmentations: rotation, color jitter, affine perspective jitter, blur
            self.transform = T.Compose([
                T.RandomRotation(degrees=(-12, 12)),
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.records[idx]
        full_path = config.PROJECT_ROOT / item["patch_path"]

        # Read as 3-channel RGB image for MobileNetV3
        img = Image.open(full_path).convert("RGB")
        tensor_img = self.transform(img)
        label = int(item["class_idx"])
        return tensor_img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for cross-entropy loss."""
        counts = Counter(r["class_idx"] for r in self.records)
        total = float(len(self.records))
        n_classes = len(config.BUBBLE_CLASSES)
        weights = []
        for i in range(n_classes):
            c = counts.get(i, 1)
            # Inverse frequency with square-root dampening
            w = np.sqrt(total / (c * n_classes))
            weights.append(w)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        # Normalize weights so mean is 1.0
        return weights_tensor / weights_tensor.mean()
