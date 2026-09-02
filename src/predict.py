"""Classify a document image with the trained model.

Loads the frozen ResNet50 backbone plus the trained linear head and reports class
probabilities. This is the demo entry point -- point it at any photo of an identity
document and it tells you whether it looks like a driving licence, a passport or an
ID card.

A caveat worth saying out loud when demoing: the model was trained on documents that
fill the frame (the `crop` variant), standing in for a pipeline where a detector
locates the document first. Feed it a wide shot of a licence on a desk and accuracy
drops -- that is the gap the `full` variant quantifies, not a bug.

Usage:
    python src/predict.py path/to/image.jpg
    python src/predict.py data/processed/crop/passport/*/TA/*.jpg --top 2
    python src/predict.py photo.jpg --variant full
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from features import PREPROCESS, build_backbone
from splits import TASK_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def letterbox_like_training(img: Image.Image, size: int = 256) -> Image.Image:
    """Match ingest.py's letterboxing so inference sees what training saw."""
    w, h = img.size
    scale = size / max(w, h)
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img.resize(new, Image.BILINEAR), ((size - new[0]) // 2, (size - new[1]) // 2))
    return canvas


def load_model(variant: str):
    head_path = MODELS_DIR / f"head_{variant}.pt"
    if not head_path.exists():
        raise SystemExit(f"{head_path} not found. Run: python src/train.py --variant {variant}")
    backbone = build_backbone()
    head = nn.Linear(2048, len(TASK_CLASSES))
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.eval()
    return backbone, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image paths (globs allowed)")
    ap.add_argument("--variant", choices=["crop", "full"], default="crop")
    ap.add_argument("--top", type=int, default=3, help="how many classes to show")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.images:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matched if matched else [Path(pattern)])
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no readable images matched")

    backbone, head = load_model(args.variant)

    with torch.inference_mode():
        for path in paths:
            img = Image.open(path).convert("RGB")
            tensor = PREPROCESS(letterbox_like_training(img)).unsqueeze(0)
            probs = torch.softmax(head(backbone(tensor))[0], dim=0)
            order = probs.argsort(descending=True)[: args.top]

            print(f"\n{path.name}")
            for rank, c in enumerate(order):
                bar = "#" * round(float(probs[c]) * 40)
                marker = "->" if rank == 0 else "  "
                print(f"  {marker} {TASK_CLASSES[c]:<17} {float(probs[c]):.3f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
