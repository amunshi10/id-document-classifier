"""Build train/val/test splits at the document-type level.

This is the file that decides whether the accuracy numbers mean anything, so the
reasoning is spelled out rather than buried.

MIDV-500 has three nested levels of near-duplication:

    class  ->  document type  ->  video clip  ->  frame

A frame-level random split is worthless: frames 11 and 12 come from the same
3-second video and are near-identical, so the test set is effectively the training
set. A clip-level split is still bad: all 10 clips of `19_esp_drvlic` show the *same
physical laminated card*, so the model can memorise that card's exact texture and
still score well without learning what a driving licence is.

Splitting at the document-type level is the only split that asks the question we
care about: shown a licence design it has never seen, does the model still call it a
licence?

The `other` class is excluded from the main task. It holds four unrelated documents
(US Social Security card, US border crossing card, US passport card, China home
return permit) that share no visual identity, and with four types a held-out type
would be a category of one. The SSN card is kept aside for qualitative demos.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.json"

TASK_CLASSES = ["driving_licence", "passport", "id_card"]
SEED = 20260902

# Roughly 70/15/15 by document type, with a floor of 2 held-out types per class so
# neither val nor test rests on a single document design.
VAL_FRAC, TEST_FRAC = 0.15, 0.15
MIN_HELD_OUT = 2


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_splits(rows: list[dict], seed: int = SEED) -> dict[str, list[str]]:
    """Partition document types into train/val/test, stratified by class."""
    types_by_class: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["label"] in TASK_CLASSES:
            types_by_class[r["label"]].add(r["doc_type"])

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for label in TASK_CLASSES:
        types = sorted(types_by_class[label])
        rng.shuffle(types)
        n = len(types)
        n_test = max(MIN_HELD_OUT, round(n * TEST_FRAC))
        n_val = max(MIN_HELD_OUT, round(n * VAL_FRAC))
        if n_test + n_val >= n:
            raise ValueError(
                f"class {label!r} has only {n} document types, too few to hold out "
                f"{n_val} val + {n_test} test and still train on something"
            )
        splits["test"] += types[:n_test]
        splits["val"] += types[n_test : n_test + n_val]
        splits["train"] += types[n_test + n_val :]

    return {k: sorted(v) for k, v in splits.items()}


def assign(rows: list[dict], splits: dict[str, list[str]]) -> dict[str, str]:
    """Map document type -> split name, for rows in the task classes."""
    lookup = {}
    for name, types in splits.items():
        for t in types:
            lookup[t] = name
    return lookup


def summarise(rows: list[dict], splits: dict[str, list[str]]) -> str:
    lookup = assign(rows, splits)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    types: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        split = lookup.get(r["doc_type"])
        if split is None:
            continue
        counts[(split, r["label"])] += 1
        types[(split, r["label"])].add(r["doc_type"])

    lines = [f"{'split':<7} {'class':<17} {'types':>6} {'frames':>8}", "-" * 42]
    for split in ("train", "val", "test"):
        for label in TASK_CLASSES:
            lines.append(
                f"{split:<7} {label:<17} {len(types[(split, label)]):>6} "
                f"{counts[(split, label)]:>8}"
            )
        lines.append("-" * 42)

    lines.append("\nHeld-out document types (never seen in training):")
    for split in ("val", "test"):
        lines.append(f"  {split}: {', '.join(splits[split])}")
    return "\n".join(lines)


def main() -> int:
    rows = load_manifest()
    splits = build_splits(rows)
    SPLIT_FILE.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(summarise(rows, splits))
    print(f"\nWrote {SPLIT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
