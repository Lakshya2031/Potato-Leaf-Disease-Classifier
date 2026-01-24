import os
import shutil
import argparse
from pathlib import Path
from typing import List

REQUIRED_CLASSES = ["Early_Blight", "Late_Blight", "Healthy"]


def ensure_structure(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for c in REQUIRED_CLASSES:
        (root / c).mkdir(exist_ok=True)


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_class_dirs(base: Path) -> List[Path]:
    found = []
    for c in REQUIRED_CLASSES:
        candidate = base / c
        if candidate.is_dir():
            found.append(candidate)
    return found


def flatten_if_nested(source: Path, target: Path) -> None:
    """If the PlantVillage subset is inside nested folders, copy class folders here."""
    if source.resolve() == target.resolve():
        return
    found = find_class_dirs(source)
    if len(found) == len(REQUIRED_CLASSES):
        for d in found:
            dest = target / d.name
            if count_images(dest) == 0:  # only copy if empty
                print(f"Copying images for class {d.name}...")
                for f in d.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(d)
                        out_path = dest / rel
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, out_path)
    else:
        print("No complete class set found to flatten.")


def verify(root: Path) -> None:
    print("Verifying dataset structure...")
    missing = [c for c in REQUIRED_CLASSES if not (root / c).is_dir()]
    if missing:
        print("Missing class folders:", ", ".join(missing))
        print("Create them or supply dataset before training.")
        return
    for c in REQUIRED_CLASSES:
        n = count_images(root / c)
        print(f"Class {c}: {n} images")
    total = sum(count_images(root / c) for c in REQUIRED_CLASSES)
    if total == 0:
        print("WARNING: No images found. Populate the folders before training.")
    else:
        print(f"Total images: {total}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and verify potato leaf disease dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Target data directory containing class folders")
    parser.add_argument("--source", type=str, default="", help="Optional source directory if dataset is nested")
    parser.add_argument("--flatten", action="store_true", help="Attempt to copy class folders from source into data-dir")
    parser.add_argument("--verify-only", action="store_true", help="Only verify structure")
    parser.add_argument("--auto-map", action="store_true", help="Map Kaggle PlantVillage potato class folder names (Potato___Early_blight etc.) into target standardized names")
    args = parser.parse_args()

    target = Path(args.data_dir)
    ensure_structure(target)

    if args.flatten and args.source:
        src = Path(args.source)
        if not src.exists():
            print(f"Source path '{src}' does not exist.")
        else:
            flatten_if_nested(src, target)

    if args.auto_map and args.source:
        src = Path(args.source)
        if not src.exists():
            print(f"Source path '{src}' does not exist for auto-map.")
        else:
            # Kaggle naming patterns
            mapping = {
                'Potato___Early_blight': 'Early_Blight',
                'Potato___Late_blight': 'Late_Blight',
                'Potato___healthy': 'Healthy'
            }
            for k, v in mapping.items():
                k_dir = src / k
                if not k_dir.is_dir():
                    print(f"Missing expected source folder: {k_dir}")
                    continue
                dest_dir = target / v
                # copy only if dest empty
                if count_images(dest_dir) == 0:
                    print(f"Mapping '{k}' -> '{v}' ...")
                    for f in k_dir.rglob('*'):
                        if f.is_file():
                            rel = f.relative_to(k_dir)
                            out_path = dest_dir / rel
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, out_path)
                else:
                    print(f"Destination {dest_dir} already has images; skipping copy.")
            print("Auto-map complete.")

    if args.verify_only:
        verify(target)
    else:
        verify(target)
        print("Done.")

if __name__ == "__main__":
    main()
