"""Assemble the S3231 dataset into the layout config.py expects.

Google Drive splits the dataset folder into several zip parts
(AI-for-Vision-Dataset-<timestamp>-1-001.zip, -002.zip, ...), each holding a
disjoint subset of files under a top-level "AI-for-Vision-Dataset/" folder.
This script merges all parts (zip files or already-extracted folders) into:

    <OMR_DATA_ROOT>/
    ├── vision_ds/            README.md, roster.csv, images/, labels/
    └── stress_test/          p03/, p04/, p19/ -> symlinks into vision_ds/images

The stress test folder contains symlinks only (no duplicated image data).
roster.csv is copied as a file and never read or printed by this script.

Usage:
    python scripts/setup_dataset.py ~/Downloads/AI-for-Vision-Dataset-*.zip
    python scripts/setup_dataset.py "path/AI-for-Vision-Dataset" "path/AI-for-Vision-Dataset 2"
    python scripts/setup_dataset.py parts... --data-root /custom/omr-data/data
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

EXPECTED_IMAGES = 2184
EXPECTED_FORMS = 740
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def _relative_member(name: str) -> PurePosixPath:
    """Strip the top-level dataset folder from an archive member path."""
    parts = PurePosixPath(name).parts
    if parts and parts[0].startswith("AI-for-Vision-Dataset"):
        parts = parts[1:]
    return PurePosixPath(*parts) if parts else PurePosixPath()


def merge_zip(zip_path: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = _relative_member(info.filename)
            if not rel.parts or rel.name in SKIP_NAMES or ".." in rel.parts or rel.parts[0] == "__MACOSX":
                continue
            target = dest / Path(*rel.parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def merge_folder(folder: Path, dest: Path, move: bool) -> int:
    count = 0
    for src in folder.rglob("*"):
        if not src.is_file() or src.name in SKIP_NAMES:
            continue
        target = dest / src.relative_to(folder)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(target))
        else:
            shutil.copy2(src, target)
        count += 1
    return count


def build_stress_links(data_root: Path) -> int:
    images_dir = data_root / "vision_ds" / "images"
    count = 0
    for person in config.EXCLUDED_PERSONS:
        src_dir = images_dir / person
        if not src_dir.is_dir():
            print(f"WARNING: {src_dir} missing; stress_test/{person} not created")
            continue
        out_dir = data_root / "stress_test" / person
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(src_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            link = out_dir / img.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path("..") / ".." / "vision_ds" / "images" / person / img.name)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Google Drive dataset parts into OMR_DATA_ROOT.")
    parser.add_argument("parts", nargs="+", type=Path, help="Zip parts and/or extracted part folders.")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT, help="Target data root.")
    parser.add_argument("--move", action="store_true", help="Move files out of extracted folders instead of copying.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    dest = data_root / "vision_ds"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Target: {dest}")

    for part in args.parts:
        if part.is_file() and part.suffix.lower() == ".zip":
            n = merge_zip(part, dest)
        elif part.is_dir():
            n = merge_folder(part, dest, move=args.move)
        else:
            print(f"Skipping {part}: not a zip file or folder")
            continue
        print(f"  {part.name}: {n} files added")

    n_images = sum(1 for p in (dest / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"})
    json_dir = dest / "labels" / "json"
    n_forms = len(list(json_dir.glob("*.json"))) if json_dir.is_dir() else 0
    print(f"vision_ds: {n_images} images (expected {EXPECTED_IMAGES}), {n_forms} label files (expected {EXPECTED_FORMS})")
    for required in ("labels/labels.csv", "labels/images.csv", "roster.csv", "README.md"):
        if not (dest / required).exists():
            print(f"WARNING: {required} missing; did you pass every zip part?")

    n_links = build_stress_links(data_root)
    print(f"stress_test: {n_links} symlinks for {', '.join(config.EXCLUDED_PERSONS)}")

    if data_root != config.DATA_ROOT:
        print(f"Set OMR_DATA_ROOT={data_root} in .env")
    ok = n_images == EXPECTED_IMAGES and n_forms == EXPECTED_FORMS
    print("Dataset ready." if ok else "Dataset incomplete; check the warnings above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
