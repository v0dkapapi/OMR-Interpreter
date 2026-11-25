"""Fine-tune TrOCR on S3231 header handwriting (name, module code, seat, date, student number).

Inputs come from scripts/build_ocr_dataset.py (cache/ocr_crops/manifest.csv),
cropped exactly as at inference. Targets are the labels.csv values; dates are
ISO (YYYY-MM-DD), so the model learns to output normalized dates directly.

Data discipline:
- Training: train-split forms only (stress contributors p03, p04, p19 excluded).
- Checkpoint selection ("dev"): val/test forms 60-99 in round-robin order.
- Forms 0-59 in that order are the OCR benchmark (scripts/evaluate_ocr.py) and
  are never used here. Val/test contributors never appear in training.

Memory: by default the image encoder and the vocabulary embedding are frozen
and Adafactor is used, so training fits in about 4 GB (0.74 s per batch of 8 on
an M5). Full fine-tuning with AdamW needs about 12 GB and swaps on a 16 GB Mac.

Only aggregate metrics are printed; names are never printed.

Usage:
    python src/train_trocr.py --epochs 6
    python src/train_trocr.py --epochs 1 --limit 200      # smoke run
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402

logger = logging.getLogger("omr.train_trocr")
CROPS_DIR = config.CACHE_DIR / "ocr_crops"
BENCHMARK_FORMS = 60
DEV_FORMS = 40


def normalize(field: str, text: str) -> str:
    text = (text or "").strip()
    if field == "name":
        return re.sub(r"\s+", " ", text.lower())
    if field in ("module_code", "seat_number"):
        return re.sub(r"[\s\-.]", "", text.upper())
    if field == "student_number_written":
        return re.sub(r"\D", "", text)
    return re.sub(r"\s", "", text)


def split_records(records: List[Dict[str, str]]):
    """Train records, dev records; benchmark forms are dropped."""
    from src.dataset import _round_robin_by_contributor

    with open(config.VISION_LABELS_DIR / "labels.csv", encoding="utf-8") as f:
        heldout_rows = [r for r in csv.DictReader(f)
                        if r["split"] in ("val", "test") and r["contributor"] not in config.EXCLUDED_PERSONS]
    ordered = [r["form_id"] for r in _round_robin_by_contributor(heldout_rows)]
    dev_forms = set(ordered[BENCHMARK_FORMS:BENCHMARK_FORMS + DEV_FORMS])
    train = [r for r in records if r["split"] == "train" and r["contributor"] not in config.EXCLUDED_PERSONS]
    dev = [r for r in records if r["form_id"] in dev_forms]
    return train, dev


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild capture-condition augmentation: rotation, scale, contrast, blur, noise."""
    h, w = img.shape[:2]
    angle = rng.uniform(-2.5, 2.5)
    scale = rng.uniform(0.92, 1.05)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:
        img = img + np.random.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, records, processor, train: bool, max_len: int = 32):
        self.records, self.processor, self.train, self.max_len = records, processor, train, max_len
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(str(CROPS_DIR / r["path"]))
        if self.train:
            img = augment(img, self.rng)
        pixel_values = self.processor(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), return_tensors="pt"
        ).pixel_values[0]
        ids = self.processor.tokenizer(r["target"], max_length=self.max_len, truncation=True).input_ids
        return pixel_values, ids


def collate(batch, pad_id: int):
    pixel_values = torch.stack([b[0] for b in batch])
    longest = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        labels[i, : len(ids)] = torch.tensor(ids)
    return pixel_values, labels


@torch.no_grad()
def evaluate(model, processor, records, device, batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    hits: Dict[str, List[bool]] = {}
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        images = [Image.fromarray(cv2.cvtColor(cv2.imread(str(CROPS_DIR / r["path"])), cv2.COLOR_BGR2RGB)) for r in chunk]
        pv = processor(images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pv, max_new_tokens=24, num_beams=1)
        texts = processor.batch_decode(out, skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            hits.setdefault(r["field"], []).append(normalize(r["field"], t) == normalize(r["field"], r["target"]))
    model.train()
    return {f: 100.0 * sum(v) / len(v) for f, v in sorted(hits.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on header handwriting.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--limit", type=int, default=None, help="Use only N training crops (smoke runs).")
    parser.add_argument("--train-encoder", action="store_true", help="Also fine-tune the image encoder (more memory).")
    parser.add_argument("--out", type=Path, default=Path(config.OCR_CONFIG["trocr_finetuned_dir"]))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    torch.manual_seed(0)
    random.seed(0)

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    with open(CROPS_DIR / "manifest.csv", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    train, dev = split_records(records)
    if args.limit:
        random.Random(0).shuffle(train)
        train = train[: args.limit]
    logger.info(f"train crops: {len(train)} | dev crops: {len(dev)} "
                f"(train contributors {len({r['contributor'] for r in train})}, dev {len({r['contributor'] for r in dev})})")

    device = config.get_device()
    base = config.OCR_CONFIG["trocr_model"]
    processor = TrOCRProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base)
    tok = processor.tokenizer
    model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.to(device)

    loader = torch.utils.data.DataLoader(
        CropDataset(train, processor, train=True), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=0,
    )
    # Freeze the vocabulary embedding (tied with the output projection) and, by
    # default, the image encoder; train the decoder layers with Adafactor.
    from transformers.optimization import Adafactor
    for p in model.decoder.get_input_embeddings().parameters():
        p.requires_grad = False
    if not args.train_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.0f}M")
    optimizer = Adafactor(trainable, lr=args.lr, scale_parameter=False, relative_step=False, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, cycle_momentum=False
    )

    base_scores = evaluate(model, processor, dev, device)
    logger.info(f"dev exact match before fine-tuning: {json.dumps({k: round(v, 1) for k, v in base_scores.items()})}")
    best = float(np.mean(list(base_scores.values())))
    history = [{"epoch": 0, "dev": base_scores}]

    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for i, (pv, labels) in enumerate(loader, 1):
            loss = model(pixel_values=pv.to(device), labels=labels.to(device)).loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if step < total_steps - 1:
                    scheduler.step()
                step += 1
            if i % 50 == 0:
                logger.info(f"epoch {epoch} batch {i}/{len(loader)} loss {running / i:.4f} ({time.time() - t0:.0f}s)")
        scores = evaluate(model, processor, dev, device)
        mean = float(np.mean(list(scores.values())))
        history.append({"epoch": epoch, "train_loss": running / len(loader), "dev": scores})
        logger.info(f"epoch {epoch} train loss {running / len(loader):.4f} | dev exact match "
                    f"{json.dumps({k: round(v, 1) for k, v in scores.items()})} | mean {mean:.1f}% ({time.time() - t0:.0f}s)")
        if mean > best:
            best = mean
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            logger.info(f"saved best checkpoint (mean dev exact {mean:.1f}%) to {args.out}")

    (config.OUTPUTS_DIR / "trocr_finetune_history.json").write_text(json.dumps(history, indent=2))
    logger.info(f"best mean dev exact match: {best:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
