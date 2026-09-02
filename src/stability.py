"""How much should you trust the 82.6%?

Re-running train.py gives the identical number every time, because the seed is fixed.
That is determinism, and it is worth confirming, but it says nothing about whether the
result is stable or a lucky draw. Two things could be varying underneath it:

  training seed   head initialisation and batch order, with the split held fixed.
                  Small variance expected -- a linear probe on frozen features is a
                  near-convex problem.

  split seed      which document designs get held out. This is the one that matters.
                  With only 12 licence / 15 passport / 19 ID types, the test set is 7
                  designs, and the score depends on which 7. If the spread here is
                  wide, no single split's number should be quoted without an error bar.

Usage:
    python src/stability.py
"""

from __future__ import annotations

import numpy as np
import torch

from splits import TASK_CLASSES, build_splits, load_manifest
from train import load_features, macro_f1, train_head

N_RUNS = 5


def evaluate(data, y, where, seed) -> tuple[float, float, dict[str, float]]:
    tr, va, te = where == "train", where == "val", where == "test"
    head = train_head(
        data["feats"][tr], y[tr], data["feats"][va], y[va],
        len(TASK_CLASSES), epochs=60, seed=seed,
    )
    head.eval()
    with torch.inference_mode():
        pred = head(torch.from_numpy(data["feats"][te])).argmax(1).numpy()
    acc = float(np.mean(y[te] == pred))
    f1 = macro_f1(y[te], pred, len(TASK_CLASSES))

    per_type = {}
    for dt in sorted(set(data["doc_type"][te])):
        m = data["doc_type"][te] == dt
        per_type[dt] = float(np.mean(y[te][m] == pred[m]))
    return acc, f1, per_type


def main() -> int:
    data = load_features("crop")
    keep = np.array([l in TASK_CLASSES for l in data["label"]])
    data = {k: v[keep] for k, v in data.items()}
    y = np.array([TASK_CLASSES.index(l) for l in data["label"]], dtype=np.int64)
    rows = [r for r in load_manifest() if r["label"] in TASK_CLASSES]

    base_splits = build_splits(rows)
    base_of = {t: s for s, ts in base_splits.items() for t in ts}
    base_where = np.array([base_of[dt] for dt in data["doc_type"]])

    print("=" * 66)
    print("A. Training seed varied, split held fixed")
    print("=" * 66)
    accs = []
    for i in range(N_RUNS):
        acc, f1, _ = evaluate(data, y, base_where, seed=1000 + i)
        accs.append(acc)
        print(f"  seed {1000 + i}: acc {acc:.4f}  macroF1 {f1:.4f}")
    print(f"\n  mean {np.mean(accs):.4f}  sd {np.std(accs):.4f}  "
          f"range [{min(accs):.4f}, {max(accs):.4f}]")

    print()
    print("=" * 66)
    print("B. Split seed varied -- different document designs held out each time")
    print("=" * 66)
    split_accs, chn_in_test = [], []
    for i in range(N_RUNS):
        sp = build_splits(rows, seed=2000 + i)
        of = {t: s for s, ts in sp.items() for t in ts}
        where = np.array([of[dt] for dt in data["doc_type"]])
        acc, f1, per_type = evaluate(data, y, where, seed=1000)
        split_accs.append(acc)
        has_chn = "09_chn_id" in sp["test"]
        chn_in_test.append(has_chn)
        marker = "  <- 09_chn_id in test" if has_chn else ""
        print(f"  split {2000 + i}: acc {acc:.4f}  macroF1 {f1:.4f}{marker}")
        print(f"      held out: {', '.join(sp['test'])}")

    print(f"\n  mean {np.mean(split_accs):.4f}  sd {np.std(split_accs):.4f}  "
          f"range [{min(split_accs):.4f}, {max(split_accs):.4f}]")

    with_chn = [a for a, h in zip(split_accs, chn_in_test) if h]
    without = [a for a, h in zip(split_accs, chn_in_test) if not h]
    if with_chn and without:
        print(f"\n  splits WITH 09_chn_id in test:    mean {np.mean(with_chn):.4f} "
              f"(n={len(with_chn)})")
        print(f"  splits WITHOUT 09_chn_id in test: mean {np.mean(without):.4f} "
              f"(n={len(without)})")
        print(f"  -> that single document type moves the headline by "
              f"{abs(np.mean(without) - np.mean(with_chn)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
