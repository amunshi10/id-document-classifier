# Identity Document Classification — ResNet50 Transfer Learning

Classifies photographs of identity documents as **driving licence**, **passport**, or
**ID card**, using an ImageNet-pretrained ResNet50 as a frozen feature extractor with a
trained linear head, in PyTorch.

Trained and evaluated on [MIDV-500](https://arxiv.org/abs/1807.05786) — 15,000 frames
from 500 mobile-phone videos of 50 identity document types. Every document in it is a
**public-domain specimen with a fictional holder**, so no real personal data is involved.
See [DATASET.md](DATASET.md) for the full data rationale.

## Headline result

**82.6% frame accuracy / 85.7% clip accuracy** on document designs the model has never
seen, against a 42.9% majority-class baseline.

That number is deliberately lower than it could be. The reason is the next section.

## The split is the whole story

MIDV-500 has three levels of near-duplication: class → document type → video clip →
frame. How you split determines whether your accuracy means anything.

| Split rule | Frame accuracy | What it actually measures |
|---|---|---|
| `random` (by frame) | 97.3% | Nothing. Frames 11 and 12 of one 3-second video land on opposite sides of the split. |
| `clip` (by video) | 97.1% | Nothing. Every test clip shows the *same physical card* as training. |
| **`doctype`** (by design) | **82.6%** | Generalisation to document designs never seen. |

Same model, same features, same training-set size (3,220 vs 3,200 frames). **Only the
split rule differs — and it is worth ~15 accuracy points of pure illusion.**

Reproduce it yourself:

```bash
python src/train.py --variant crop --split-mode random
```

The held-out test designs are Albanian ID, Azerbaijani passport, Chinese ID, Spanish
licence, Greek passport, Italian licence, Slovak ID. None appear in training.

## The interesting failure

The 82.6% headline hides a bimodal outcome:

| Held-out design | Accuracy |
|---|---|
| Italian licence | 99% |
| Spanish licence | 98% |
| Greek passport | 96% |
| Slovak ID | 96% |
| Albanian ID | 95% |
| Azerbaijani passport | 94% |
| **Chinese ID** | **0%** |

Six of seven designs land at 94–99%. One is at zero, and it accounts for 109 of the 122
total errors. Excluding it, the model is at **96.3%**.

The cause is visible immediately in `reports/all_id_cards.jpg`: of the 19 ID-card types
in MIDV-500, eighteen show a **portrait photo**. `09_chn_id` is the reverse face —
national emblem and title text, no portrait, no personal-data fields. The model learned
"ID card ⇒ contains a portrait," which is true for 18 of 19 types and false for this one,
so it falls through to the card-shaped alternative: driving licence.

This is a **dataset-composition finding, not a bug**. Fixing it means adding non-portrait
document faces to training, not tuning hyperparameters — and it is only discoverable
under the document-type split. A random split memorises the Chinese ID and reports 97%.

### Testing that explanation instead of asserting it

"It looks like the portrait matters" is a hypothesis. `src/occlusion.py` tests it by
sliding a grey patch across correctly-classified ID cards and measuring where
`P(id_card)` drops:

| Document | P(id_card) | under worst occlusion | pushed toward |
|---|---|---|---|
| Slovak ID | 0.560 | 0.357 | `driving_licence` |
| Albanian ID | 0.427 | 0.333 | `driving_licence` |
| Chinese ID | 0.240 | 0.199 | `driving_licence` (already misclassified) |

`reports/occlusion.jpg` shows the sensitivity concentrated over the **portrait region**
on both cards the model gets right. Occluding it reproduces the Chinese ID failure on
documents the model normally classifies correctly — which is the causal claim, not just
a correlation with something visible in the image.

### Why the Chinese ID is not excluded from the test set

It is a legitimate, correctly-labelled national ID card. Dropping a test case because the
model fails on it inflates the headline number by deleting the evidence — 96.3% earned
that way is not a real 96.3%. It stays in, the failure is diagnosed, and the fix is
named. That is more defensible than a cleaner number with a hole in it.

## Pipeline

```
ingest.py  ->  splits.py  ->  features.py  ->  train.py  ->  predict.py
```

| Stage | What it does | Time |
|---|---|---|
| `ingest.py` | Streams 50 MIDV-500 archives: download → perspective-rectify → 256px JPEG → delete. Peak disk ~1.5 GB instead of 33 GB. | ~75 min |
| `splits.py` | Partitions document *types* into train/val/test, stratified by class. | instant |
| `features.py` | Caches frozen ResNet50 2048-d embeddings once. | ~11 min |
| `train.py` | Trains the linear head on cached features; val-selected, test touched once. | seconds |
| `predict.py` | Classifies any document image. | instant |

### Why frozen features rather than full fine-tuning

This was developed on a laptop with no CUDA GPU. Fine-tuning all 25M ResNet50 parameters
would take hours per epoch; embedding every image once and training a linear head takes
**11 minutes then seconds per run**, which means model selection is affordable rather
than aspirational.

The trade-off, stated honestly: cached embeddings mean no random augmentation during head
training. For a linear probe on a frozen backbone that costs little.

The resulting model is a 2048×3 linear layer — **26 KB**.

## Preprocessing: two variants

Each frame is rendered twice, so the value of document localisation is measured rather
than assumed:

- **`crop`** — the document perspective-rectified out of the frame using MIDV-500's
  ground-truth corner quadrangle. Stands in for a production pipeline with a detector
  upstream.
- **`full`** — the whole camera frame. What a classifier sees with no detector; the
  document is often under 15% of the pixels.

Both are letterboxed rather than squashed, because aspect ratio is real signal — a licence
is card-shaped, an open passport nearly square.

See `reports/rectify_check.jpg` for the rectification working across all five capture
conditions.

### What the detector is worth

Running the identical model on both variants answers the question with a measurement:

| Variant | Frame accuracy | Clip accuracy | Macro F1 |
|---|---|---|---|
| `crop` (detector upstream) | **82.6%** | **85.7%** | 0.837 |
| `full` (no detector) | 73.9% | 77.1% | 0.743 |

**Document localisation is worth about 9 accuracy points.** That is the cost of skipping
the detection stage, and it is a design decision with a number attached rather than an
assumption.

The degradation is not uniform. Passport recall barely moves (0.985 on `full`) — booklet
shape survives being small in frame. The Italian licence collapses from 99% to 65%,
because fine card layout is what disappears when the document is 15% of the pixels.

## Results detail

Under the document-type split, `crop` variant:

```
frame accuracy    0.8257  (578/700)
macro F1          0.8370
majority baseline 0.4286
clip accuracy     0.8571  (60/70)     <- majority vote over a clip's frames

per-class          prec  recall     n
driving_licence   0.629   0.985   200
passport          1.000   0.950   200
id_card           0.970   0.637   300
```

Driving-licence precision (0.629) is depressed almost entirely by the Chinese ID cards
flooding into that class. Passport precision is perfect — booklets are shape-separable
from cards.

Robustness across capture conditions is flat, which is the encouraging part:

| Condition | Accuracy |
|---|---|
| table | 85.7% |
| clutter | 85.0% |
| hand | 85.0% |
| keyboard | 84.3% |
| partial | 72.9% |

Only `partial` — where the camera pans off the document — degrades meaningfully.

## Reproducing

```bash
pip install torch torchvision pillow numpy
python src/ingest.py
python src/splits.py
python src/features.py --variant crop
python src/train.py --variant crop
```

The dataset is not committed (~420 MB); `ingest.py` rebuilds it from the FTP source.

## Demo

```bash
python src/predict.py path/to/document.jpg
```

On the held-out Spanish licence:

```
TA19_02.jpg
  -> driving_licence   0.554  ######################
     id_card           0.257  ##########
     passport          0.189  ########
```

## Known limitations

- **No reject option.** The head has three outputs and always picks one. Shown a US
  Social Security card — which belongs to the excluded `other` class — it returns
  `driving_licence` at 0.49. A production system needs a confidence threshold or an
  explicit "not a recognised document" class.
- **One document face per type.** MIDV-500 captures a single side of each document, so
  the model never sees the reverse of an ID card. This is precisely what breaks it on
  `09_chn_id`.
- **`other` is excluded from the task.** Four unrelated documents (SSN card, border
  crossing card, passport card, home return permit) share no visual identity, and a
  held-out type would be a category of one.
- **No augmentation during head training**, a direct consequence of caching embeddings.
  A fine-tuning run with augmentation is the natural next comparison.

## Next steps

1. Add a confidence threshold and an `unknown` class to handle out-of-distribution input.
2. Fine-tune `layer4` with augmentation and compare against the linear probe.
3. Evaluate on [MIDV-2019](ftp://smartengines.com/midv-500/extra/midv-2019/) — the same
   documents under low light and extreme projective distortion — as a robustness test.

## Licence and attribution

MIDV-500 is distributed under **CC BY-SA 2.5**. If you publish results or images from it,
cite:

> Arlazarov, V.V., Bulatov, K., Chernov, T., Arlazarov, V.L. *MIDV-500: A Dataset for
> Identity Document Analysis and Recognition on Mobile Devices in Video Stream.*
> Computer Optics 43(5), 2019.
