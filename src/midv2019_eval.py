"""Does the model survive strong projective distortion and low light?

MIDV-2019 re-shoots the same 50 document designs under two conditions MIDV-500 never
covered -- severe projective distortion and low lighting -- on newer 4K phone cameras.
Because the designs are identical, no retraining is needed or wanted: the six
cross-validation heads trained on MIDV-500 are reused unchanged, and MIDV-2019 acts
purely as a harder test set.

That gives a 2x2 nobody in this project has been able to fill in before:

                    easy conditions      hard conditions
                    (MIDV-500)           (MIDV-2019)
  known design      ~96%                 ?     <- pure capture-condition effect
  unseen design     ~71%                 ?     <- conditions and generalisation together

The top row is the valuable one. Those are designs the fold trained on, so the *only*
thing that changes is how the photograph was taken. Nothing else in this repository
isolates robustness from generalisation.

The trap this script exists to avoid: pooling all MIDV-2019 frames against one model
would silently mix designs that model trained on with designs it did not, producing a
number that answers neither question. Fold membership is respected throughout -- for
each fold, a MIDV-2019 frame is "known" if that fold trained on its design and "unseen"
if it was held out.

Usage:
    python src/midv2019_eval.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from crossval import make_folds
from splits import TASK_CLASSES
from train import load_features, train_head

SEED = 20260902


def load_2019(variant: str) -> dict:
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "models" / f"features_midv2019_{variant}.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n"
            f"  python src/ingest.py --dataset midv2019\n"
            f"  python src/features.py --dataset midv2019 --variant {variant} --stride 1"
        )
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def acc(mask, y_true, y_pred) -> tuple[float, int]:
    if not mask.any():
        return float("nan"), 0
    return float(np.mean(y_true[mask] == y_pred[mask])), int(mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--variant", choices=["crop", "full"], default="crop")
    args = ap.parse_args()

    # MIDV-500: what the heads are trained on.
    d500 = load_features(args.variant)
    keep = np.array([l in TASK_CLASSES for l in d500["label"]])
    d500 = {k: v[keep] for k, v in d500.items()}
    y500 = np.array([TASK_CLASSES.index(l) for l in d500["label"]], dtype=np.int64)

    # MIDV-2019: evaluation only, never trained on.
    d19 = load_2019(args.variant)
    keep19 = np.array([l in TASK_CLASSES for l in d19["label"]])
    d19 = {k: v[keep19] for k, v in d19.items()}
    y19 = np.array([TASK_CLASSES.index(l) for l in d19["label"]], dtype=np.int64)

    print(f"MIDV-500 {len(y500)} frames (training) | MIDV-2019 {len(y19)} frames (eval only)")
    print(f"MIDV-2019 conditions: {sorted(set(d19['condition']))}")
    print(f"MIDV-2019 devices:    {sorted(set(d19['device']))}\n")

    types_by_class: dict[str, list[str]] = defaultdict(list)
    for dt, lbl in zip(d500["doc_type"], d500["label"]):
        if dt not in types_by_class[lbl]:
            types_by_class[lbl].append(dt)
    folds = make_folds(types_by_class, args.folds, SEED)

    # Accumulate per-frame correctness, tagged by whether the fold had seen the design.
    known_hits, unseen_hits = [], []
    cond_hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    dev_hits: dict[str, list[int]] = defaultdict(list)
    base_known, base_unseen = [], []

    for i in range(args.folds):
        test_types, val_types = set(folds[i]), set(folds[(i + 1) % args.folds])
        where = np.array([
            "test" if dt in test_types else "val" if dt in val_types else "train"
            for dt in d500["doc_type"]
        ])
        tr, va, te = where == "train", where == "val", where == "test"

        head = train_head(d500["feats"][tr], y500[tr], d500["feats"][va], y500[va],
                          len(TASK_CLASSES), epochs=60, seed=SEED)
        head.eval()

        # MIDV-500 baselines for this fold, for the left column of the 2x2.
        with torch.inference_mode():
            p500 = head(torch.from_numpy(d500["feats"])).argmax(1).numpy()
        base_known.append((y500[tr] == p500[tr]).astype(int))
        base_unseen.append((y500[te] == p500[te]).astype(int))

        # MIDV-2019, same heads, split by whether this fold trained on the design.
        with torch.inference_mode():
            p19 = head(torch.from_numpy(d19["feats"])).argmax(1).numpy()
        correct19 = (y19 == p19).astype(int)

        train_types = {dt for dt in d500["doc_type"][tr]}
        is_known = np.array([dt in train_types for dt in d19["doc_type"]])
        is_unseen = np.array([dt in test_types for dt in d19["doc_type"]])

        known_hits.append(correct19[is_known])
        unseen_hits.append(correct19[is_unseen])

        for cond in sorted(set(d19["condition"])):
            m = (d19["condition"] == cond) & is_known
            cond_hits[(cond, "known")].extend(correct19[m].tolist())
            m = (d19["condition"] == cond) & is_unseen
            cond_hits[(cond, "unseen")].extend(correct19[m].tolist())
        for dev in sorted(set(d19["device"])):
            m = (d19["device"] == dev) & is_known
            dev_hits[dev].extend(correct19[m].tolist())

        print(f"  fold {i} done")

    k500 = np.concatenate(base_known).mean()
    u500 = np.concatenate(base_unseen).mean()
    k19 = np.concatenate(known_hits).mean()
    u19 = np.concatenate(unseen_hits).mean()

    print("\n" + "=" * 70)
    print("ROBUSTNESS: capture conditions vs design generalisation")
    print("=" * 70)
    print(f"  {'':<16} {'MIDV-500':>12} {'MIDV-2019':>12} {'drop':>10}")
    print(f"  {'':<16} {'(easy)':>12} {'(hard)':>12}")
    print("  " + "-" * 52)
    # Shown unsigned: the column is labelled "drop", so a leading + would read as a gain.
    print(f"  {'known design':<16} {k500:>11.1%} {k19:>12.1%} {k500 - k19:>9.1%} pts")
    print(f"  {'unseen design':<16} {u500:>11.1%} {u19:>12.1%} {u500 - u19:>9.1%} pts")
    print()
    print("  Note: the MIDV-500 'known design' cell is training-set accuracy and is")
    print("  optimistic by construction. The honest comparison for that row is the drop,")
    print("  not the level -- both cells contain designs the model was fitted on, so the")
    print("  difference between them isolates the capture conditions.")

    print("\n" + "=" * 70)
    print("BY CONDITION (MIDV-2019 only)")
    print("=" * 70)
    for regime in ("known", "unseen"):
        print(f"  {regime} designs:")
        for cond in sorted(set(d19["condition"])):
            h = cond_hits[(cond, regime)]
            if h:
                print(f"    {cond:<12} {np.mean(h):.4f}  (n={len(h)})")

    print("\n" + "=" * 70)
    print("BY CAMERA (known designs)")
    print("=" * 70)
    for dev, h in sorted(dev_hits.items()):
        if h:
            print(f"  {dev:<16} {np.mean(h):.4f}  (n={len(h)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
