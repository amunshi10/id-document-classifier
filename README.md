# Identity Document Classification — ResNet50 Transfer Learning

Classifies photographs of identity documents as **driving licence**, **passport**, or
**ID card**, using an ImageNet-pretrained ResNet50 as a frozen feature extractor with a
trained linear head, in PyTorch.

Trained and evaluated on [MIDV-500](https://arxiv.org/abs/1807.05786) — 15,000 frames
from 500 mobile-phone videos of 50 identity document types. Every document in it is a
**public-domain specimen with a fictional holder**, so no real personal data is involved.
See [DATASET.md](DATASET.md) for the full data rationale.

## Headline result

**70.8% ± 5.7% accuracy**, cross-validated across six folds of held-out document designs,
against a ~43% majority-class baseline.

The error bar is the point. An earlier version of this README reported **82.6%** from a
single train/test split. That number was real but misleading, and the story of how it got
corrected is the most useful thing in this repository.

## Why one split was not enough

The test set in a single split holds 700 frames — which sounds comfortable until you
notice they come from only **7 document designs**. The independent unit is the design,
not the frame, so the effective sample size is 7, not 700.

`src/stability.py` measures what that costs by varying two things independently:

| What varies | Spread in accuracy |
|---|---|
| Training seed (head init, batch order), split fixed | **± 0.003** |
| Which document designs are held out | **± 0.103**, range 0.571 – 0.853 |

Model training is essentially deterministic. **The split is where all the variance lives**,
and it is 30× larger. A single split's number is one draw from a wide distribution, and
the original 82.6% happened to land on the favourable end — it drew a test set containing
only one design the model is bad at, where the average draw contains two.

`src/crossval.py` does the appropriate thing instead: partition all 46 document types into
six stratified folds, hold each out in turn, report mean and spread.

```
fold 0: 0.7133    fold 3: 0.6414
fold 1: 0.8150    fold 4: 0.7357
fold 2: 0.6587    fold 5: 0.6843

CROSS-VALIDATED ACCURACY  0.7081 +/- 0.0572
```

**The lesson generalises past this project:** when your data has few independent units,
report an interval, not a point. A single split will happily hand you a number 12 points
too high and no indication anything is wrong.

## The split rule is worth ~26 points of illusion

This finding survived cross-validation and is the most important one here. MIDV-500 has
three levels of near-duplication — class → document type → video clip → frame — and how
you split determines whether your accuracy means anything at all.

| Split rule | Accuracy | What it actually measures |
|---|---|---|
| `random` (by frame) | 97.3% | Nothing. Frames 11 and 12 of one 3-second video land on opposite sides. |
| `clip` (by video) | 97.1% | Nothing. Every test clip shows the *same physical card* as training. |
| **`doctype`, cross-validated** | **70.8%** | Generalisation to document designs never seen. |

Same model, same features, near-identical training-set size. Reproduce it:

```bash
python src/train.py --variant crop --split-mode random
python src/crossval.py --folds 6
```

## Generalising to unseen designs is genuinely hard

The cross-validated per-design scores show the real picture — this is bimodal, not a
uniform 71%:

```
29_irn_drvlic    0.000     27_hrv_passport  1.000
09_chn_id        0.010     20_esp_id_new    1.000
14_deu_id_new    0.010     06_bra_passport  1.000
26_hrv_drvlic    0.030     25_grc_passport  0.980
35_nor_drvlic    0.110     30_ita_drvlic    0.980
```

**13 of 46 document types score below 0.60 when held out.** Roughly two-thirds of designs
transfer well; the rest fail outright. This is the honest characterisation of the model,
and it is invisible under a single split — which reports one number and tells you nothing
about which designs are behind it.

### One failure diagnosed causally

`09_chn_id` (0.01) is worth singling out because the cause is identifiable. Of the 19
ID-card types, eighteen show a **portrait photo**; `09_chn_id` is the emblem face — no
portrait, no personal-data fields. The model appears to have learned "ID card ⇒ contains
a portrait" and falls through to the card-shaped alternative, driving licence.

`src/occlusion.py` tests that rather than asserting it, by sliding a grey patch across
correctly-classified ID cards and measuring where `P(id_card)` drops:

| Document | P(id_card) | under worst occlusion | pushed toward |
|---|---|---|---|
| Slovak ID | 0.560 | 0.357 | `driving_licence` |
| Albanian ID | 0.427 | 0.333 | `driving_licence` |

`reports/occlusion.jpg` shows the sensitivity concentrated over the portrait region.
Masking it **reproduces the Chinese ID failure on documents the model normally gets
right** — a causal claim, not a correlation with something visible in the image.

The failing designs stay in the evaluation. Dropping a test case because the model fails
on it inflates the headline by deleting the evidence.

## What document localisation is worth: not established

Each frame is rendered twice, so the value of a detection stage could be measured rather
than assumed:

- **`crop`** — document perspective-rectified out of the frame using MIDV-500's
  ground-truth corner quadrangle. Stands in for a pipeline with a detector upstream.
- **`full`** — the whole camera frame; the document is often under 15% of the pixels.

| Variant | Cross-validated accuracy |
|---|---|
| `crop` | 0.708 ± 0.057 |
| `full` | 0.687 ± 0.053 |

A single split showed `crop` ahead by 9 points, which looked decisive. Across six folds
the gap is **+2.2 points with a paired SD of 6.8** (t = 0.78 on 5 df; 4 of 6 folds favour
`crop`). **That is not a significant difference**, and the 9-point version was an artifact
of one favourable split.

The honest conclusion is that this experiment does not have the statistical power to
resolve the question — 6 folds over 46 document types is too few. Reporting it as a
9-point win would have been wrong.

## Pipeline

```
ingest.py -> splits.py -> features.py -> train.py -> predict.py
                                      \-> crossval.py   (headline number)
                                      \-> stability.py  (why CV is needed)
                                      \-> occlusion.py  (failure diagnosis)
```

| Stage | What it does | Time |
|---|---|---|
| `ingest.py` | Streams 50 archives: download → perspective-rectify → 256px JPEG → delete. Peak disk ~1.5 GB instead of 33 GB. | ~75 min |
| `features.py` | Caches frozen ResNet50 2048-d embeddings once. | ~11 min |
| `crossval.py` | Six-fold CV across document types. | ~1 min |
| `predict.py` | Classifies any document image. | instant |

### Why frozen features rather than full fine-tuning

Developed on a laptop with no CUDA GPU. Fine-tuning all 25M ResNet50 parameters would take
hours per epoch; embedding every image once and training a linear head takes **11 minutes
then seconds per run** — which is exactly what made the stability analysis affordable. A
setup where each run cost an hour would have shipped the 82.6% and never caught it.

Trade-off, stated plainly: cached embeddings mean no augmentation during head training.
The resulting model is a 2048×3 linear layer — **26 KB**.

## Robustness across capture conditions

Flat, which is the encouraging part (single split, `crop`):

| Condition | Accuracy |
|---|---|
| table | 85.7% |
| clutter | 85.0% |
| hand | 85.0% |
| keyboard | 84.3% |
| partial | 72.9% |

Only `partial` — camera panning off the document — degrades meaningfully.

## Known limitations

- **No reject option.** Three outputs, always picks one. Shown a US Social Security card
  (excluded `other` class) it returns `driving_licence` at 0.49.
- **A third of designs don't transfer.** 13 of 46 score below 0.60 when unseen.
- **One document face per type.** MIDV-500 captures a single side, so the model never sees
  the reverse of an ID card — precisely what breaks `09_chn_id`.
- **Underpowered ablations.** 46 document types in 6 folds cannot resolve differences
  smaller than roughly 10 points.
- **No augmentation** during head training, a consequence of caching embeddings.

## Reproducing

```bash
pip install torch torchvision pillow numpy
python src/ingest.py
python src/features.py --variant crop
python src/crossval.py --folds 6
```

The dataset is not committed (~420 MB); `ingest.py` rebuilds it from the FTP source.

## Licence and attribution

MIDV-500 is distributed under **CC BY-SA 2.5**. If you publish results or images from it,
cite:

> Arlazarov, V.V., Bulatov, K., Chernov, T., Arlazarov, V.L. *MIDV-500: A Dataset for
> Identity Document Analysis and Recognition on Mobile Devices in Video Stream.*
> Computer Optics 43(5), 2019.
