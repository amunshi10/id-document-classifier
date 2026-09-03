"""Give the model a way to say "I don't know".

The classifier has three outputs and always picks one. Shown a US Social Security card
-- a document type it was never trained on -- it returns `driving_licence` at 0.49. It
is not just wrong, it is wrong without warning, and that is the single limitation that
makes the 70.8% unseen-design regime unsafe to deploy.

This adds selective prediction: abstain when the maximum softmax probability falls below
a threshold. Two things are then worth measuring, and they are different questions:

  error detection   among documents of the three trained classes, does low confidence
                    actually correlate with being wrong? If yes, abstaining raises
                    accuracy on what remains.

  OOD detection     shown a document from the excluded `other` class (SSN card, US
                    border-crossing card, US passport card, China home-return permit),
                    does the model abstain rather than confidently guessing?

Methodology note. The threshold is chosen on the validation fold using **only the three
trained classes**, then applied unchanged to the test fold and to the OOD set. The
`other` frames never influence the threshold. Tuning on them and then reporting
rejection rates on them would be circular, and would overstate the result.

Baseline is max softmax probability (Hendrycks & Gimpel, 2017) -- the standard reference
point, not a sophisticated method. Reporting it honestly is the point.

Usage:
    python src/reject.py --target 0.90
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from crossval import make_folds
from splits import TASK_CLASSES
from train import load_features, train_head

SEED = 20260902


def confidence(head, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class, max softmax probability)."""
    head.eval()
    with torch.inference_mode():
        probs = torch.softmax(head(torch.from_numpy(feats)), dim=1).numpy()
    return probs.argmax(1), probs.max(1)


def mahalanobis_scorer(feats_tr: np.ndarray, y_tr: np.ndarray, n_classes: int):
    """Class-conditional Mahalanobis distance in feature space (Lee et al., 2018).

    Max softmax only sees the 3 logits, which is a very lossy summary -- an out-of-
    distribution document can land confidently inside a decision region because the head
    has nowhere else to put it. This instead asks whether the 2048-d embedding is
    anywhere near the training distribution at all.

    Covariance is estimated with Ledoit-Wolf shrinkage: 2048 dimensions from ~3000
    samples is badly under-determined, and the raw sample covariance would be singular.
    """
    from sklearn.covariance import LedoitWolf

    mus = np.stack([feats_tr[y_tr == c].mean(0) for c in range(n_classes)])
    centred = np.concatenate([feats_tr[y_tr == c] - mus[c] for c in range(n_classes)])
    prec = LedoitWolf(assume_centered=True).fit(centred).precision_

    def score(feats: np.ndarray) -> np.ndarray:
        # Higher = more in-distribution, to match the sign convention of max-softmax.
        d = np.stack([
            np.einsum("ij,jk,ik->i", feats - mus[c], prec, feats - mus[c])
            for c in range(n_classes)
        ])
        return -d.min(0)

    return score


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest threshold whose selective accuracy on validation reaches `target`.

    Lowest, not highest: among thresholds that meet the accuracy target we want the one
    that abstains least, since every abstention is a document a human now has to look at.
    """
    for tau in np.arange(0.34, 1.00, 0.005):
        keep = conf >= tau
        if keep.sum() < 20:            # too few kept for the estimate to mean anything
            break
        if correct[keep].mean() >= target:
            return float(tau)
    return 1.01                        # target unreachable: abstain on everything


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--variant", choices=["crop", "full"], default="crop")
    ap.add_argument("--skip-maha", action="store_true",
                    help="skip the slow Mahalanobis comparison")
    ap.add_argument("--target", type=float, default=0.90,
                    help="selective accuracy to aim for, tuned on validation")
    args = ap.parse_args()

    data = load_features(args.variant)
    is_task = np.array([l in TASK_CLASSES for l in data["label"]])
    ood = {k: v[~is_task] for k, v in data.items()}      # the `other` class
    task = {k: v[is_task] for k, v in data.items()}
    y = np.array([TASK_CLASSES.index(l) for l in task["label"]], dtype=np.int64)

    print(f"{is_task.sum()} in-distribution frames | {(~is_task).sum()} OOD frames "
          f"({', '.join(sorted(set(ood['doc_type'])))})\n")

    types_by_class: dict[str, list[str]] = defaultdict(list)
    for dt, lbl in zip(task["doc_type"], task["label"]):
        if dt not in types_by_class[lbl]:
            types_by_class[lbl].append(dt)
    folds = make_folds(types_by_class, args.folds, SEED)

    all_conf, all_correct, all_kept, taus = [], [], [], []
    ood_conf, ood_kept = [], []
    maha_id, maha_ood = [], []

    for i in range(args.folds):
        test_types, val_types = set(folds[i]), set(folds[(i + 1) % args.folds])
        where = np.array([
            "test" if dt in test_types else "val" if dt in val_types else "train"
            for dt in task["doc_type"]
        ])
        tr, va, te = where == "train", where == "val", where == "test"

        head = train_head(task["feats"][tr], y[tr], task["feats"][va], y[va],
                          len(TASK_CLASSES), epochs=60, seed=SEED)

        # Threshold chosen on validation, in-distribution only. Never sees OOD.
        vp, vc = confidence(head, task["feats"][va])
        tau = pick_threshold(vc, (vp == y[va]).astype(float), args.target)
        taus.append(tau)

        tp, tc = confidence(head, task["feats"][te])
        all_conf.append(tc)
        all_correct.append((tp == y[te]).astype(int))
        all_kept.append((tc >= tau).astype(int))

        _, oc = confidence(head, ood["feats"])
        ood_conf.append(oc)
        ood_kept.append((oc >= tau).astype(int))

        if not args.skip_maha:
            # Second scorer, fitted on the same training fold, for comparison.
            maha = mahalanobis_scorer(task["feats"][tr], y[tr], len(TASK_CLASSES))
            maha_id.append(maha(task["feats"][te]))
            maha_ood.append(maha(ood["feats"]))

        print(f"  fold {i}: tau={tau:.3f}")

    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct)
    kept = np.concatenate(all_kept).astype(bool)
    oconf = np.concatenate(ood_conf)
    okept = np.concatenate(ood_kept).astype(bool)

    print("\n" + "=" * 68)
    print("HOW WELL DOES CONFIDENCE RANK ANYTHING?  (AUROC, 0.5 = useless)")
    print("=" * 68)
    auc_err = roc_auc_score(correct, conf)
    auc_ood = roc_auc_score(
        np.r_[np.ones(len(conf)), np.zeros(len(oconf))], np.r_[conf, oconf]
    )
    print(f"  max-softmax, correct vs incorrect (error detection)     {auc_err:.3f}")
    print(f"  max-softmax, in-distribution vs OOD (novelty detection) {auc_ood:.3f}")

    if not maha_id:
        print("  (Mahalanobis skipped)")
        mid = mood = None
    if maha_id:
        mid, mood = np.concatenate(maha_id), np.concatenate(maha_ood)
        auc_m_err = roc_auc_score(correct, mid)
        auc_m_ood = roc_auc_score(
            np.r_[np.ones(len(mid)), np.zeros(len(mood))], np.r_[mid, mood]
        )
        print(f"  Mahalanobis, correct vs incorrect                       {auc_m_err:.3f}")
        print(f"  Mahalanobis, in-distribution vs OOD                     {auc_m_ood:.3f}")

    print("\n" + "=" * 68)
    print("THRESHOLD SWEEP (pooled over folds)")
    print("=" * 68)
    print(f"  {'tau':>5} {'coverage':>9} {'sel.acc':>9} {'OOD kept':>9}   "
          f"{'<- lower is better':>0}")
    for tau in (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        k = conf >= tau
        cov = k.mean()
        acc = correct[k].mean() if k.any() else float("nan")
        ok = (oconf >= tau).mean()
        print(f"  {tau:>5.2f} {cov:>8.1%} {acc:>9.1%} {ok:>9.1%}")

    print("\n" + "=" * 68)
    print(f"OPERATING POINT (tau tuned per fold on validation, target {args.target:.0%})")
    print("=" * 68)
    print(f"  mean tau across folds        {np.mean(taus):.3f}")
    print(f"  coverage (accepted)          {kept.mean():.1%}")
    print(f"  accuracy on accepted         {correct[kept].mean():.1%}"
          if kept.any() else "  nothing accepted")
    print(f"  accuracy without abstention  {correct.mean():.1%}")
    print(f"  OOD documents wrongly accepted {okept.mean():.1%}")
    print(f"  OOD documents correctly rejected {1 - okept.mean():.1%}")

    print()
    print("=" * 68)
    print("REJECTION BY OOD DOCUMENT TYPE")
    print("=" * 68)
    print("  Not all 'other' documents are equally out of distribution. A US passport")
    print("  card is visually an ID card; only the SSN card looks unlike anything the")
    print("  model was trained on. If the score means anything, that should show here.")
    ood_types = np.tile(ood["doc_type"], args.folds)
    for dt in sorted(set(ood_types.tolist())):
        m = ood_types == dt
        print(f"  {dt:<26} rejected {1 - okept[m].mean():.1%}  (n={m.sum()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
