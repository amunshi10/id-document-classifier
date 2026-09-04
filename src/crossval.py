"""Cross-validate across document types, because a single split is not enough evidence.

The test set has 700 frames, which sounds comfortable until you notice they come from
only **7 document designs**. The independent unit here is the design, not the frame, so
the effective sample size is 7 -- and `stability.py` shows what that does: varying which
designs are held out swings accuracy from 0.57 to 0.85, while varying the training seed
moves it by 0.003.

Quoting one split's number as *the* result is therefore quoting one draw from a wide
distribution. This script does the statistically appropriate thing instead: partition all
46 document types into K folds, hold each fold out in turn, and report mean and spread.

Usage:
    python src/crossval.py --folds 6
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np
import torch

from splits import TASK_CLASSES
from train import load_features, macro_f1, train_head

SEED = 20260902


def make_folds(types_by_class: dict[str, list[str]], k: int, seed: int) -> list[list[str]]:
    """Stratified fold assignment: each fold gets a share of every class."""
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for label in sorted(types_by_class):
        types = sorted(types_by_class[label])
        rng.shuffle(types)
        for i, t in enumerate(types):
            folds[i % k].append(t)
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--variant", choices=["crop", "full"], default="crop")
    ap.add_argument("--hidden", type=int, default=0,
                    help="hidden units; 0 = linear probe (the default)")
    args = ap.parse_args()

    data = load_features(args.variant)
    keep = np.array([l in TASK_CLASSES for l in data["label"]])
    data = {k: v[keep] for k, v in data.items()}
    y = np.array([TASK_CLASSES.index(l) for l in data["label"]], dtype=np.int64)

    types_by_class: dict[str, list[str]] = defaultdict(list)
    for dt, lbl in zip(data["doc_type"], data["label"]):
        if dt not in types_by_class[lbl]:
            types_by_class[lbl].append(dt)

    folds = make_folds(types_by_class, args.folds, SEED)
    print(f"{sum(len(f) for f in folds)} document types across {args.folds} folds "
          f"| variant={args.variant}\n")

    accs, f1s, per_type_all = [], [], {}
    cond_hits: dict[str, list[int]] = defaultdict(list)
    for i in range(args.folds):
        test_types = set(folds[i])
        val_types = set(folds[(i + 1) % args.folds])
        where = np.array([
            "test" if dt in test_types else "val" if dt in val_types else "train"
            for dt in data["doc_type"]
        ])
        tr, va, te = where == "train", where == "val", where == "test"

        head = train_head(
            data["feats"][tr], y[tr], data["feats"][va], y[va],
            len(TASK_CLASSES), epochs=60, seed=SEED, hidden=args.hidden,
        )
        head.eval()
        with torch.inference_mode():
            pred = head(torch.from_numpy(data["feats"][te])).argmax(1).numpy()

        acc = float(np.mean(y[te] == pred))
        f1 = macro_f1(y[te], pred, len(TASK_CLASSES))
        accs.append(acc)
        f1s.append(f1)

        for dt in sorted(test_types):
            m = data["doc_type"][te] == dt
            if m.any():
                per_type_all[dt] = float(np.mean(y[te][m] == pred[m]))

        # Pool correctness per capture condition across folds. Every frame is in
        # exactly one test fold, so this ends up covering the whole dataset once.
        correct = (y[te] == pred).astype(int)
        for cond, ok in zip(data["condition"][te], correct):
            cond_hits[str(cond)].append(int(ok))

        print(f"fold {i}: acc {acc:.4f}  macroF1 {f1:.4f}  ({len(test_types)} types held out)")

    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATED ACCURACY  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"CROSS-VALIDATED MACRO F1  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"range [{min(accs):.4f}, {max(accs):.4f}]")
    print("=" * 60)

    print("\nPer capture condition, pooled over all folds:")
    for cond in sorted(cond_hits, key=lambda c: -np.mean(cond_hits[c])):
        hits = cond_hits[cond]
        print(f"  {cond:<12} {np.mean(hits):.4f}  (n={len(hits)})")

    print("\nEvery document type, scored while held out:")
    for dt, a in sorted(per_type_all.items(), key=lambda kv: kv[1]):
        bar = "#" * round(a * 30)
        flag = "  <-- weak" if a < 0.6 else ""
        print(f"  {dt:<26} {a:.3f}  {bar}{flag}")

    weak = [d for d, a in per_type_all.items() if a < 0.6]
    print(f"\n{len(weak)} of {len(per_type_all)} document types score below 0.60 when unseen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
