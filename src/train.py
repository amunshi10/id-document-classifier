"""Train and evaluate the document-type classification head on cached ResNet50 features.

Everything here runs on the (N, 2048) matrices produced by features.py, so a full
training run takes seconds and we can afford honest model selection: tune on val,
touch test exactly once at the end.

Reported metrics, and why each one is here:

  frame accuracy      the headline number, on document designs never seen in training
  macro F1            the classes are imbalanced (19 ID types vs 12 licence types),
                      so plain accuracy flatters the majority class
  clip accuracy       majority vote over the ~30 frames of a clip. This is what a real
                      system does -- you get a video, not a single frame -- and it is
                      the number worth quoting for a production pipeline
  per-condition       does it hold up when the document is in a hand or half out of
                      frame, or only on a clean table?
  per-doc-type        which specific unseen designs it fails on, which is where the
                      interesting failure analysis lives

Usage:
    python src/train.py --variant crop
    python src/train.py --variant full          # the no-detector comparison
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from splits import SPLIT_FILE, TASK_CLASSES, build_splits, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SEED = 20260902


def assign_split(data: dict, split_of: dict[str, str], mode: str) -> np.ndarray:
    """Assign every frame to train/val/test under one of three splitting rules.

    `doctype` is the real one -- entire document designs are held out, so the test
    set contains passports and licences whose layout the model has never seen.

    `clip` and `random` are here to be beaten, not used. They exist so the leakage
    argument in DATASET.md is a measurement rather than an assertion:

      random  splits individual frames. Frames 11 and 12 of the same 3-second video
              land on opposite sides of the split, so the test set is effectively a
              copy of the training set.
      clip    splits whole video clips but keeps document types together, so every
              test clip shows the same physical card the model trained on. Better
              than random, still not measuring generalisation.

    All three keep the same 70/15/15 proportions, so the accuracy gap between them
    is attributable to the leak and not to training-set size.
    """
    if mode == "doctype":
        return np.array([split_of[dt] for dt in data["doc_type"]])

    if mode == "clip":
        keys = np.array([f"{d}/{c}" for d, c in zip(data["doc_type"], data["clip"])])
    else:
        keys = np.arange(len(data["doc_type"])).astype(str)

    unique = sorted(set(keys.tolist()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    n_test = round(len(unique) * 0.15)
    n_val = round(len(unique) * 0.15)
    bucket = {}
    for i, k in enumerate(unique):
        bucket[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return np.array([bucket[k] for k in keys])


def load_features(variant: str):
    path = MODELS_DIR / f"features_{variant}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python src/features.py --variant {variant}")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def get_splits() -> dict[str, list[str]]:
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    splits = build_splits(load_manifest())
    SPLIT_FILE.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def train_head(Xtr, ytr, Xva, yva, n_classes, epochs=60, lr=3e-4, wd=1e-4, seed=SEED):
    """Linear probe on frozen features, selected on val macro-F1."""
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], n_classes)

    # Class weighting: ID cards outnumber licences ~19:12 by document type, and a head
    # trained without this quietly learns to prefer the majority class.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)))

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=weights)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)

    best_f1, best_state, best_epoch = -1.0, None, -1
    n = len(Xtr_t)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        head.eval()
        with torch.inference_mode():
            pred = head(Xva_t).argmax(1).numpy()
        f1 = macro_f1(yva, pred, n_classes)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    print(f"  best val macro-F1 {best_f1:.4f} at epoch {best_epoch}")
    return head


def macro_f1(y_true, y_pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def confusion(y_true, y_pred, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def report(y_true, y_pred, classes, meta, out: list[str]) -> None:
    n = len(classes)
    acc = float(np.mean(y_true == y_pred))
    out.append(f"\nframe accuracy   {acc:.4f}  ({np.sum(y_true == y_pred)}/{len(y_true)})")
    out.append(f"macro F1         {macro_f1(y_true, y_pred, n):.4f}")

    majority = Counter(y_true).most_common(1)[0][1] / len(y_true)
    out.append(f"majority baseline {majority:.4f}")

    out.append("\nper-class:")
    out.append(f"  {'class':<17} {'prec':>6} {'recall':>7} {'n':>6}")
    for c, name in enumerate(classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(f"  {name:<17} {prec:>6.3f} {rec:>7.3f} {tp + fn:>6}")

    out.append("\nconfusion matrix (rows = true, cols = predicted):")
    out.append("  " + " " * 17 + "".join(f"{c[:12]:>14}" for c in classes))
    for i, name in enumerate(confusion(y_true, y_pred, n)):
        out.append(f"  {classes[i]:<17}" + "".join(f"{v:>14}" for v in name))

    # Clip-level majority vote -- what a real system does with a video.
    clip_key = [f"{d}/{c}" for d, c in zip(meta["doc_type"], meta["clip"])]
    votes: dict[str, list[int]] = defaultdict(list)
    truth: dict[str, int] = {}
    for k, t, p in zip(clip_key, y_true, y_pred):
        votes[k].append(p)
        truth[k] = t
    clip_correct = sum(
        Counter(v).most_common(1)[0][0] == truth[k] for k, v in votes.items()
    )
    out.append(
        f"\nclip accuracy (majority vote over ~30 frames)  "
        f"{clip_correct / len(votes):.4f}  ({clip_correct}/{len(votes)})"
    )

    out.append("\nper capture condition:")
    for cond in sorted(set(meta["condition"])):
        m = meta["condition"] == cond
        out.append(f"  {cond:<12} {np.mean(y_true[m] == y_pred[m]):.4f}  (n={m.sum()})")

    out.append("\nper held-out document type:")
    for dt in sorted(set(meta["doc_type"])):
        m = meta["doc_type"] == dt
        a = np.mean(y_true[m] == y_pred[m])
        flag = "   <-- weak" if a < 0.7 else ""
        out.append(f"  {dt:<26} {a:.4f}  (n={m.sum()}){flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["crop", "full"], default="crop")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument(
        "--split-mode", choices=["doctype", "clip", "random"], default="doctype",
        help="doctype is the honest one; clip and random exist to demonstrate leakage",
    )
    args = ap.parse_args()

    data = load_features(args.variant)
    splits = get_splits()
    split_of = {t: s for s, ts in splits.items() for t in ts}

    keep = np.array([
        lbl in TASK_CLASSES and dt in split_of
        for lbl, dt in zip(data["label"], data["doc_type"])
    ])
    for k in data:
        data[k] = data[k][keep]

    y = np.array([TASK_CLASSES.index(l) for l in data["label"]], dtype=np.int64)
    where = assign_split(data, split_of, args.split_mode)

    tr, va, te = where == "train", where == "val", where == "test"
    print(f"variant={args.variant}  train={tr.sum()}  val={va.sum()}  test={te.sum()}")
    print(f"held-out test document types: {', '.join(splits['test'])}\n")

    head = train_head(
        data["feats"][tr], y[tr], data["feats"][va], y[va],
        len(TASK_CLASSES), epochs=args.epochs,
    )

    head.eval()
    with torch.inference_mode():
        pred = head(torch.from_numpy(data["feats"][te])).argmax(1).numpy()

    meta = {k: data[k][te] for k in ("doc_type", "clip", "condition", "device")}
    out = [
        f"MIDV-500 document classification -- variant={args.variant}",
        f"ResNet50 (ImageNet, frozen) + linear head",
        f"Split mode: {args.split_mode}",
        "=" * 64,
    ]
    report(y[te], pred, TASK_CLASSES, meta, out)

    text = "\n".join(out)
    print(text)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"results_{args.variant}_{args.split_mode}.txt").write_text(text, encoding="utf-8")
    torch.save(head.state_dict(), MODELS_DIR / (f"head_{args.variant}.pt" if args.split_mode == "doctype" else f"head_{args.variant}_{args.split_mode}.pt"))
    print(f"\nWrote reports/results_{args.variant}.txt and models/head_{args.variant}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
