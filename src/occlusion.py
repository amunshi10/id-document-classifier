"""Occlusion sensitivity: test *why* the model calls something an ID card.

The `09_chn_id` failure came with a hypothesis -- the model learned "ID card contains a
portrait photo", so the emblem face of the Chinese ID falls through to the card-shaped
alternative, driving licence. Looking at pictures makes that plausible. It does not make
it true.

This tests it directly. Slide a grey patch across a correctly-classified ID card and
measure how much P(id_card) drops at each position. If the hypothesis holds, the drop
concentrates over the portrait, and occluding it flips the prediction to driving_licence
-- reproducing the Chinese ID failure on a document the model normally gets right.

If instead the sensitivity is spread evenly, or sits on the text or the card border, the
hypothesis is wrong and the README needs correcting.

Usage:
    python src/occlusion.py
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from pathlib import Path

from features import PREPROCESS
from predict import load_model
from splits import TASK_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed" / "crop"
REPORTS = PROJECT_ROOT / "reports"

PATCH = 64      # grey square edge, px, on the 256px image
STRIDE = 16
GREY = 128


def sensitivity(backbone, head, img: Image.Image, target: int):
    """Return (heatmap, baseline_probs, flipped_to) for one image."""
    with torch.inference_mode():
        base = torch.softmax(head(backbone(PREPROCESS(img).unsqueeze(0)))[0], 0).numpy()

    arr = np.array(img)
    positions, batch = [], []
    for y in range(0, 256 - PATCH + 1, STRIDE):
        for x in range(0, 256 - PATCH + 1, STRIDE):
            patched = arr.copy()
            patched[y : y + PATCH, x : x + PATCH] = GREY
            batch.append(PREPROCESS(Image.fromarray(patched)))
            positions.append((y, x))

    probs = []
    with torch.inference_mode():
        for i in range(0, len(batch), 64):
            chunk = torch.stack(batch[i : i + 64])
            probs.append(torch.softmax(head(backbone(chunk)), 1).numpy())
    probs = np.concatenate(probs)

    heat = np.zeros((256, 256), dtype=np.float32)
    counts = np.zeros((256, 256), dtype=np.float32)
    for (y, x), p in zip(positions, probs):
        heat[y : y + PATCH, x : x + PATCH] += base[target] - p[target]
        counts[y : y + PATCH, x : x + PATCH] += 1
    heat /= np.maximum(counts, 1)

    # Which class does the worst occlusion push us into?
    worst = probs[:, target].argmin()
    flipped_to = TASK_CLASSES[int(probs[worst].argmax())]
    return heat, base, flipped_to, float(probs[worst][target])


def colourise(heat: np.ndarray) -> Image.Image:
    """Red where occluding hurts the target class, blue where it helps."""
    m = float(np.abs(heat).max()) or 1.0
    norm = heat / m
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[..., 0] = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    rgb[..., 2] = (np.clip(-norm, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def main() -> int:
    backbone, head = load_model("crop")
    target = TASK_CLASSES.index("id_card")

    # Held-out ID cards the model classifies correctly, plus the one it fails on.
    picks = [
        ("id_card", "42_svk_id"),
        ("id_card", "01_alb_id"),
        ("id_card", "09_chn_id"),
    ]

    tiles = []
    for label, dt in picks:
        path = sorted((PROC / label / dt).rglob("TA*.jpg"))[0]
        img = Image.open(path).convert("RGB")
        heat, base, flipped, low = sensitivity(backbone, head, img, target)
        tiles.append((dt, img, heat))
        print(
            f"{dt:<14} P(id_card)={base[target]:.3f} -> {low:.3f} under worst occlusion"
            f"  | predicted {TASK_CLASSES[int(base.argmax())]:<15}"
            f" | worst-occlusion class: {flipped}"
        )

    out = Image.new("RGB", (256 * len(tiles), 256 * 2 + 20), (18, 18, 18))
    for i, (dt, img, heat) in enumerate(tiles):
        out.paste(img, (i * 256, 20))
        out.paste(Image.blend(img, colourise(heat), 0.6), (i * 256, 256 + 20))
    out.save(REPORTS / "occlusion.jpg", quality=93)
    print(f"\nWrote {REPORTS / 'occlusion.jpg'} (top: image, bottom: red = "
          f"occluding here reduces P(id_card))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
