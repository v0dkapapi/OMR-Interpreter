"""MobileNetV3-Small bubble classifier training script with classical baseline.

Features:
- Fine-tunes torchvision.models.mobilenet_v3_small on cropped bubble patches
- Splits train and validation strictly by PERSON (excluding p03, p04, p19)
- Handles class imbalance via inverse class weighting
- Implements data augmentations (rotation, jitter, blur, scale)
- Saves per-epoch checkpoints for resumable training
- Generates metrics report and confusion matrix (JSON and PNG)
- Includes classical fill-ratio baseline and flags model vs baseline disagreements
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix

# Ensure MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.dataset import PatchExtractor, BubbleDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_fill_ratio(patch_gray: np.ndarray, radius: int = 14) -> float:
    """Calculate the classical circular fill ratio inside a bubble patch."""
    h, w = patch_gray.shape[:2]
    cx, cy = w // 2, h // 2

    # Create circular mask
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    circle_mask = dist_sq <= (radius ** 2)

    total_circle_pixels = float(np.sum(circle_mask))
    if total_circle_pixels == 0:
        return 0.0

    # In normalized grayscale, paper is bright (~220-255) and marks are dark (< 175)
    dark_pixels = np.sum((patch_gray < 175) & circle_mask)
    return float(dark_pixels / total_circle_pixels)


def classical_classify_bubble(
    patch_gray: np.ndarray,
    fill_thresh: float = config.CLASSICAL_FILL_THRESHOLD,
    ambiguous_lower: float = config.CLASSICAL_AMBIGUOUS_LOWER,
    ambiguous_upper: float = config.CLASSICAL_AMBIGUOUS_UPPER
) -> Tuple[str, float]:
    """Rule-based classical bubble classifier based on fill ratio."""
    ratio = compute_fill_ratio(patch_gray)
    if ratio >= fill_thresh:
        return "filled", ratio
    elif ambiguous_lower <= ratio < fill_thresh:
        return "ambiguous", ratio
    else:
        return "unfilled", ratio


def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Instantiate MobileNetV3-Small with customized 3-class classification head."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier head: MobileNetV3 small classifier has:
    # (0): Linear(576, 1024), (1): Hardswish(), (2): Dropout(0.2), (3): Linear(1024, 1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def draw_confusion_matrix_image(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path
):
    """Render and save a clear confusion matrix image using OpenCV."""
    img_size = 500
    canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    # Header title
    cv2.putText(canvas, "Bubble Classification Confusion Matrix", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (17, 24, 39), 2)

    n = len(class_names)
    grid_start_x, grid_start_y = 120, 80
    cell_w, cell_h = 100, 100

    max_val = np.max(cm) if np.max(cm) > 0 else 1

    for i in range(n):
        # Row label (True class)
        cv2.putText(canvas, class_names[i], (15, grid_start_y + i * cell_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 29, 29), 2)
        # Col label (Predicted class)
        cv2.putText(canvas, class_names[i], (grid_start_x + i * cell_w + 15, grid_start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 29, 29), 2)

        for j in range(n):
            val = cm[i, j]
            # Color intensity in maroon/red palette (#7F1D1D)
            norm_val = float(val) / max_val
            # Intensity: white to dark maroon (RGB: 127, 29, 29)
            b = int(255 - norm_val * (255 - 29))
            g = int(255 - norm_val * (255 - 29))
            r = int(255 - norm_val * (255 - 127))

            top_left = (grid_start_x + j * cell_w, grid_start_y + i * cell_h)
            bot_right = (top_left[0] + cell_w, top_left[1] + cell_h)
            cv2.rectangle(canvas, top_left, bot_right, (b, g, r), -1)
            cv2.rectangle(canvas, top_left, bot_right, (229, 231, 235), 1)

            text_color = (255, 255, 255) if norm_val > 0.5 else (17, 24, 39)
            val_str = str(val)
            (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = top_left[0] + (cell_w - tw) // 2
            ty = top_left[1] + (cell_h + th) // 2
            cv2.putText(canvas, val_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.putText(canvas, "Horizontal: Predicted | Vertical: True", (100, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def train_bubble_model(
    epochs: int = 4,
    subset_size: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_forms: int = 15,
    force_extract: bool = False,
    resume: bool = False,
    checkpoint_dir: Path = config.TRAIN_CONFIG["checkpoint_dir"],
    final_model_path: Path = config.TRAIN_CONFIG["final_model_path"]
) -> Dict[str, Any]:
    """Train and evaluate the bubble patch classifier."""
    device = config.get_device()
    logger.info(f"Target compute device: {device}")

    # 1. Extract and Cache Patches
    extractor = PatchExtractor()
    records = extractor.build_cache(max_forms=max_forms, force_rebuild=force_extract)

    if not records:
        raise RuntimeError("No patch records available to train on.")

    # 2. Strict Person-based split
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] in ("val", "test")]

    # If either split is empty, partition by person
    if not train_records or not val_records:
        people = sorted(list(set(r["person"] for r in records)))
        n_train_people = max(1, int(len(people) * 0.75))
        train_people = set(people[:n_train_people])
        train_records = [r for r in records if r["person"] in train_people]
        val_records = [r for r in records if r["person"] not in train_people]

    # Verify no contributor overlap between train and val
    train_persons = set(r["person"] for r in train_records)
    val_persons = set(r["person"] for r in val_records)
    overlap = train_persons.intersection(val_persons)
    assert not overlap, f"Security violation: person overlap between train and val: {overlap}"

    # Double-check: ensure NO excluded persons in train or val records
    for r in records:
        assert r["person"] not in config.EXCLUDED_PERSONS, (
            f"Security violation: {r['person']} found in records!"
        )

    # Subsample if subset_size requested (allocate proportionally up to subset_size total)
    if subset_size and (len(train_records) + len(val_records)) > subset_size:
        max_train = min(len(train_records), int(subset_size * 0.72))
        max_val = min(len(val_records), subset_size - max_train)

        np.random.seed(42)
        idx_train = np.random.permutation(len(train_records))[:max_train]
        train_records = [train_records[i] for i in idx_train]

        np.random.seed(42)
        idx_val = np.random.permutation(len(val_records))[:max_val]
        val_records = [val_records[i] for i in idx_val]

    logger.info(
        f"Training patches: {len(train_records)} (contributors: {train_persons}) | "
        f"Validation patches: {len(val_records)} (contributors: {val_persons}) | "
        f"Total patches: {len(train_records) + len(val_records)}"
    )

    # 3. Create Datasets and DataLoaders
    train_dataset = BubbleDataset(train_records, is_train=True)
    val_dataset = BubbleDataset(val_records, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 4. Model, Loss, Optimizer
    model = build_model(num_classes=3, pretrained=True).to(device)
    class_weights = train_dataset.get_class_weights().to(device)
    logger.info(f"Class weights: {class_weights.tolist()} for classes: {config.BUBBLE_CLASSES}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Checkpoint handling: clean for fresh run unless resume=True
    start_epoch = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for old_ckpt in checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            old_ckpt.unlink()
    else:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info(f"Found checkpoint: {latest.name}, loading state...")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resuming training from epoch {start_epoch + 1}/{epochs}")

    # 5. Training Loop
    training_history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / float(total)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_correct / float(val_total) if val_total > 0 else 0.0

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # Save per-epoch checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)

    # 6. Save final model weights
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model weights saved to: {final_model_path}")

    # 7. Comprehensive Evaluation & Confusion Matrix
    model.eval()
    all_preds = []
    all_labels = []
    all_classical_preds = []
    disagreements = 0

    with torch.no_grad():
        for i, r in enumerate(val_records):
            patch_path = config.PROJECT_ROOT / r["patch_path"]
            img_bgr = cv2.imread(str(patch_path))
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

            # Classical baseline prediction
            class_pred_str, fill_ratio = classical_classify_bubble(gray)
            class_pred_idx = config.CLASS_TO_IDX[class_pred_str]
            all_classical_preds.append(class_pred_idx)

            # Neural model prediction
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = val_dataset.transform(pil_img).unsqueeze(0).to(device)
            out = model(tensor)
            pred_idx = out.argmax(dim=1).item()

            gt_label = int(r["class_idx"])
            all_preds.append(pred_idx)
            all_labels.append(gt_label)

            if pred_idx != class_pred_idx:
                disagreements += 1

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    report = classification_report(
        all_labels, all_preds, target_names=config.BUBBLE_CLASSES, output_dict=True, zero_division=0
    )

    metrics_result = {
        "final_val_accuracy": round(float(np.mean(np.array(all_preds) == np.array(all_labels))), 4),
        "classical_baseline_accuracy": round(float(np.mean(np.array(all_classical_preds) == np.array(all_labels))), 4),
        "model_vs_classical_disagreements": disagreements,
        "disagreement_rate": round(disagreements / float(len(val_records)), 4) if val_records else 0.0,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "training_history": training_history,
        "total_val_samples": len(val_records)
    }

    # Save metrics JSON
    metrics_path = config.OUTPUTS_DIR / "bubble_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)

    # Render confusion matrix image
    cm_img_path = config.OUTPUTS_DIR / "confusion_matrix.png"
    draw_confusion_matrix_image(cm, config.BUBBLE_CLASSES, cm_img_path)
    logger.info(f"Saved metrics report to {metrics_path} and confusion matrix to {cm_img_path}")

    return metrics_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 bubble classifier.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--subset", type=int, default=5000, help="Subset size for training/validation (up to N)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-forms", type=int, default=15, help="Number of forms to extract patches from")
    parser.add_argument("--force-extract", action="store_true", help="Force patch cache rebuild")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    train_bubble_model(
        epochs=args.epochs,
        subset_size=args.subset,
        batch_size=args.batch_size,
        max_forms=args.max_forms,
        force_extract=args.force_extract,
        resume=args.resume
    )
