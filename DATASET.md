# Dataset: MIDV-500

## Why this dataset

The project brief was "classify driving licences vs. social security cards." There is
no public dataset containing both, and building one by scraping real identity documents
is not an option — those are real people's personal data.

MIDV-500 solves this properly. Every source document in it is a **public-domain
specimen** (fictional holders — the Spanish licence is issued to "JAMES BOND"), sourced
from Wikimedia Commons, printed on photo paper, laminated, and then filmed on real
phones. It is the dataset the identity-document research community actually uses.

- **Paper:** Arlazarov et al., *MIDV-500: A Dataset for Identity Document Analysis and
  Recognition on Mobile Devices in Video Stream*, Computer Optics 43(5), 2019.
  [arXiv:1807.05786](https://arxiv.org/abs/1807.05786)
- **Download:** `ftp://smartengines.com/midv-500/`
- **Licence:** Creative Commons Attribution-ShareAlike 2.5 Generic (CC BY-SA 2.5)

## What's in it

50 document types × 10 video clips × ~31 frames = **15,600 frames** at 1080×1920.

Each clip is one capture condition on one phone:

| Code | Condition | | Code | Device |
|---|---|---|---|---|
| `T` | on a table | | `A` | Apple iPhone 5 |
| `K` | on a keyboard | | `S` | Samsung Galaxy S3 |
| `H` | held in hand | | | |
| `P` | partially out of frame | | | |
| `C` | cluttered background | | | |

So `HS19_10` is frame 10 of the Spanish licence, held in hand, shot on the Galaxy S3.

Every frame also carries a ground-truth **quadrangle** marking the document's four
corners in the image.

## Class mapping

The original brief's two classes become four, derived from the 50 document types:

| Class | Types | Examples |
|---|---|---|
| `driving_licence` | 12 | Spain, Germany (old + new), Japan, Italy, Norway, Iran… |
| `passport` | 15 | Brazil, Czechia, Greece, Latvia, Russia (internal)… |
| `id_card` | 19 | Albania, Chile, China, Estonia, Turkey, Ukraine… |
| `other` | 4 | **US Social Security card**, US border crossing card, US passport card, China home-return permit |

Note the original SSN card *does* exist here (`49_usa_ssn82`) — it just can't carry its
own class, because with a single document instance the model would memorise that one
card rather than learn a category. It sits in `other` and makes a good demo image.

Two judgement calls worth flagging: `39_rus_internalpassport` is classed as a passport
(it is a booklet and named one), while `48_usa_passportcard` is `other` (card-shaped,
not a booklet).

## Preprocessing

`src/ingest.py` streams one archive at a time — download, extract, render, delete — so
peak disk stays ~1.5 GB instead of the 33 GB the full TIF set would need. It writes two
variants of every frame at 256×256:

- **`full/`** — the whole camera frame, letterboxed. What a classifier sees with no
  document detector in front of it. The document is often <15% of the pixels.
- **`crop/`** — the document perspective-rectified out of the frame using the GT
  quadrangle, then letterboxed. Stands in for a production pipeline where a detector
  runs first.

Both are letterboxed rather than squashed, because aspect ratio is real signal: a licence
is card-shaped, an open passport is nearly square. Keeping both variants lets us
*measure* what document localisation is worth instead of assuming it.

See `reports/rectify_check.jpg` for a visual check of the rectification.

## The split — read this before trusting any accuracy number

Two traps, both of which produce fake high accuracy:

**1. Don't split frames randomly.** Consecutive frames come from the same 3-second video
and are near-duplicates. A random split puts frame 11 in train and frame 12 in test, and
you have effectively tested on the training set.

**2. Don't split clips randomly either.** Every clip of a document type shows the *same
physical card*. Splitting by clip still lets the model memorise that specific card
instead of learning what a licence is.

**The split we use is by document type.** Entire document designs are held out — train on
the Spanish, German and Japanese licences, test on the Norwegian and Italian ones the
model has never seen. That measures the thing we actually care about: does it generalise
to a document design it was not trained on?

Capture condition is used as a secondary axis for robustness reporting (how much worse is
`partial` than `table`?), not as the primary split.

## Attribution

CC BY-SA 2.5 requires attribution. If you publish results or images from this dataset,
cite the paper above; per-document attribution for the original Wikimedia source images
is in `sources-index.pdf` on the FTP server.

## Optional extension: MIDV-2019

`ftp://smartengines.com/midv-500/extra/midv-2019/` re-captures the same 50 document types
under **low light** and **high projective distortion**. Wrong choice as a training set,
but a good held-out robustness test: train on MIDV-500, report the degradation on
MIDV-2019.
