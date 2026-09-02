"""Cache frozen ResNet50 embeddings for every frame.

This machine has no CUDA GPU (Intel Iris Xe integrated graphics), so fine-tuning all
25M ResNet50 parameters would take hours per epoch. It also is not what transfer
learning calls for on a dataset this size.

Instead we use the standard fixed-feature-extractor setup: run every image through
the ImageNet-pretrained backbone exactly once, keep the 2048-d vector from the layer
below the classifier, and throw the rest away. Training the head then operates on a
(N, 2048) float matrix and takes seconds, which means we can afford to actually tune
it rather than accept the first run.

The trade-off, stated plainly: cached features mean no random augmentation during
head training, since every image is embedded once in one fixed form. For a linear
head on a frozen backbone that costs little. `train.py --finetune` exists for the
comparison run where augmentation does matter.

Usage:
    python src/features.py --variant crop
    python src/features.py --variant full
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from splits import load_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Images are already 256x256 letterboxed on disk. Resize the whole thing to 224
# rather than centre-cropping: a centre crop would slice the black bars off unevenly
# and clip document edges, and edge shape is part of what distinguishes these classes.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], variant: str):
        self.root = PROC_DIR / variant
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.rows[i]["path"]).convert("RGB")
        return PREPROCESS(img), i


def subsample(rows: list[dict], stride: int) -> list[dict]:
    """Keep every Nth frame within each clip.

    A clip is 30 frames cut from 3 seconds of video, so neighbouring frames are
    near-identical. Embedding all 30 costs 3x the compute of embedding 10 and adds
    almost no information -- the redundancy is real, not a sampling artefact we
    should preserve. Subsampling within the clip (rather than dropping whole clips)
    keeps every capture condition represented.
    """
    if stride <= 1:
        return rows
    by_clip: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_clip[(r["doc_type"], r["clip"])].append(r)
    kept = []
    for _, group in sorted(by_clip.items()):
        group.sort(key=lambda r: r["frame"])
        kept.extend(group[::stride])
    print(f"subsampled {len(rows)} -> {len(kept)} frames (stride {stride})")
    return kept


def build_backbone() -> torch.nn.Module:
    """ImageNet-pretrained ResNet50 with the classifier head removed."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()   # 2048-d avgpool output passes straight through
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["crop", "full"], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--stride", type=int, default=3,
        help="keep every Nth frame within each clip (default 3)",
    )
    args = ap.parse_args()

    rows = load_manifest()
    if args.variant == "crop":
        # Some 'partial' frames pan off the document entirely and have no crop.
        rows = [r for r in rows if r["has_crop"] == "1"]
    rows = subsample(rows, args.stride)
    missing = [r for r in rows if not (PROC_DIR / args.variant / r["path"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest rows have no {args.variant} image on disk "
            f"(first: {missing[0]['path']}). Has ingest.py finished?"
        )

    print(f"{len(rows)} frames | variant={args.variant} | threads={torch.get_num_threads()}")

    model = build_backbone()
    loader = DataLoader(
        FrameDataset(rows, args.variant),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    feats = np.zeros((len(rows), 2048), dtype=np.float32)
    done = 0
    start = time.time()

    with torch.inference_mode():
        for batch, idx in loader:
            out = model(batch)
            feats[idx.numpy()] = out.numpy()
            done += len(idx)
            if done % (args.batch_size * 20) < args.batch_size:
                rate = done / (time.time() - start)
                eta = (len(rows) - done) / rate / 60
                print(f"  {done}/{len(rows)}  {rate:.1f} img/s  eta {eta:.1f} min", flush=True)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"features_{args.variant}.npz"
    np.savez_compressed(
        out_path,
        feats=feats,
        path=np.array([r["path"] for r in rows]),
        doc_type=np.array([r["doc_type"] for r in rows]),
        label=np.array([r["label"] for r in rows]),
        clip=np.array([r["clip"] for r in rows]),
        condition=np.array([r["condition"] for r in rows]),
        device=np.array([r["device"] for r in rows]),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path} ({size_mb:.0f} MB) in {(time.time() - start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
