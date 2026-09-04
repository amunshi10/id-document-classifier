"""Download MIDV-500 and turn it into a compact, model-ready image dataset.

MIDV-500 ships one ~600 MB zip per document type, each holding 312 full-resolution
TIF frames (1080x1920) plus a per-frame ground-truth quadrangle marking where the
document sits in the frame. That is ~33 GB of TIF we do not want on disk.

This script streams one archive at a time: download -> extract -> render two 256px
JPEG variants per frame -> delete the archive. Peak disk stays around 1.5 GB and the
final dataset lands at a few hundred MB.

Two variants are written for every frame:

  full/  the whole camera frame, letterboxed to 256x256. This is what a classifier
         sees with no document detector in front of it -- mostly table, hand or
         keyboard, with the document somewhere inside.

  crop/  the document perspective-rectified out of the frame using the ground-truth
         quadrangle, then letterboxed to 256x256. This stands in for a production
         pipeline where a detector locates the document before classification.

Keeping both lets us measure what document localisation is actually worth, instead
of assuming it.

Usage:
    python src/ingest.py                 # all 50 document types
    python src/ingest.py --limit 6       # quick smoke test
    python src/ingest.py --only 19 49    # specific document type numbers
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

OUT_SIZE = 256
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Two datasets, same 50 document types, different capture conditions -- and different
# archive layouts, which is the part that bites. MIDV-500 nests everything under a
# per-document folder (`01_alb_id/images/TA/TA01_01.tif`); MIDV-2019 does not
# (`images/LG/LG01_01.tif`). A frame filter written for one silently matches nothing
# on the other, so the depth is part of the config rather than assumed.
DATASETS = {
    "midv500": {
        "ftp": "ftp://smartengines.com/midv-500/dataset",
        "proc": "processed",
        "depth": 3,
        "conditions": {"T": "table", "K": "keyboard", "H": "hand",
                       "P": "partial", "C": "clutter"},
        "devices": {"A": "iphone5", "S": "galaxy_s3"},
    },
    "midv2019": {
        "ftp": "ftp://smartengines.com/midv-500/extra/midv-2019/dataset",
        "proc": "midv2019",
        "depth": 2,
        "conditions": {"D": "distorted", "L": "lowlight"},
        "devices": {"X": "iphone_xs_max", "G": "galaxy_s10"},
    },
}

# The 50 MIDV-500 document type codes, exactly as they appear on the FTP server.
DOC_TYPES = [
    "01_alb_id", "02_aut_drvlic_new", "03_aut_id_old", "04_aut_id", "05_aze_passport",
    "06_bra_passport", "07_chl_id", "08_chn_homereturn", "09_chn_id", "10_cze_id",
    "11_cze_passport", "12_deu_drvlic_new", "13_deu_drvlic_old", "14_deu_id_new",
    "15_deu_id_old", "16_deu_passport_new", "17_deu_passport_old", "18_dza_passport",
    "19_esp_drvlic", "20_esp_id_new", "21_esp_id_old", "22_est_id", "23_fin_drvlic",
    "24_fin_id", "25_grc_passport", "26_hrv_drvlic", "27_hrv_passport", "28_hun_passport",
    "29_irn_drvlic", "30_ita_drvlic", "31_jpn_drvlic", "32_lva_passport", "33_mac_id",
    "34_mda_passport", "35_nor_drvlic", "36_pol_drvlic", "37_prt_id", "38_rou_drvlic",
    "39_rus_internalpassport", "40_srb_id", "41_srb_passport", "42_svk_id", "43_tur_id",
    "44_ukr_id", "45_ukr_passport", "46_ury_passport", "47_usa_bordercrossing",
    "48_usa_passportcard", "49_usa_ssn82", "50_xpo_id",
]

# Class assignment. The MIDV-500 paper groups these as ID cards, passports, driving
# licences and "other"; we spell the mapping out rather than regexing the codes so
# the edge cases are visible and arguable.
#
# Deliberate calls worth knowing about:
#   39_rus_internalpassport -> passport (it is a booklet, named a passport)
#   48_usa_passportcard     -> other (card-shaped, not a passport booklet)
#   49_usa_ssn82            -> other (the US Social Security card)
#   08_chn_homereturn       -> other (travel permit)
#   47_usa_bordercrossing   -> other
CLASS_OF = {}
for _code in DOC_TYPES:
    if "drvlic" in _code:
        CLASS_OF[_code] = "driving_licence"
    elif "passport" in _code and _code != "48_usa_passportcard":
        CLASS_OF[_code] = "passport"
    elif _code.endswith("_id") or "_id_" in _code:
        CLASS_OF[_code] = "id_card"
    else:
        CLASS_OF[_code] = "other"

CONDITION_NAMES = {
    "T": "table", "K": "keyboard", "H": "hand", "P": "partial", "C": "clutter",
}
DEVICE_NAMES = {"A": "iphone5", "S": "galaxy_s3"}


def letterbox(img: Image.Image, size: int = OUT_SIZE) -> Image.Image:
    """Resize onto a square canvas without distorting aspect ratio.

    Aspect ratio is a genuine signal here -- a licence is card-shaped, an open
    passport is nearly square -- so squashing everything to a square would throw
    away information the model should be allowed to use.
    """
    w, h = img.size
    scale = size / max(w, h)
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = img.resize(new, Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return canvas


def quad_side_lengths(quad):
    """Average width and height of a quadrangle given as TL, TR, BR, BL."""
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = quad

    def dist(ax, ay, bx, by):
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    width = (dist(tlx, tly, trx, try_) + dist(blx, bly, brx, bry)) / 2
    height = (dist(tlx, tly, blx, bly) + dist(trx, try_, brx, bry)) / 2
    return width, height


def rectify(img: Image.Image, quad) -> Image.Image | None:
    """Perspective-rectify the document out of the frame using its GT quadrangle.

    Returns None when the document is essentially not in the frame, which happens
    in the 'partial' capture condition where the camera pans off the document.
    """
    width, height = quad_side_lengths(quad)
    if width < 40 or height < 40:
        return None

    # Keep the document's own proportions, capped so a sliver of a document at a
    # sharp angle cannot demand an enormous output buffer.
    ratio = width / height
    if ratio >= 1:
        out_w, out_h = OUT_SIZE, max(1, min(OUT_SIZE, round(OUT_SIZE / ratio)))
    else:
        out_w, out_h = max(1, min(OUT_SIZE, round(OUT_SIZE * ratio))), OUT_SIZE

    # PIL's QUAD transform wants the source corners as NW, SW, SE, NE, while MIDV
    # stores them as TL, TR, BR, BL. Reorder, do not just flatten.
    tl, tr, br, bl = quad
    data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
    out = img.transform((out_w, out_h), Image.QUAD, data, Image.BILINEAR)
    return letterbox(out)


def download(code: str, dest: Path, ftp_root: str, attempts: int = 3) -> bool:
    """Fetch one archive, retrying transient network failures.

    A 39 GB run over FTP will hit failures that have nothing to do with the data:
    DNS blips ("could not resolve host"), connection resets, stalled transfers.
    Losing an archive to one of those and only discovering it at the end wastes the
    whole run, so retry a few times with backoff before giving up. Permanent errors
    (a 404 on a code that does not exist) fail all attempts quickly anyway.
    """
    url = f"{ftp_root}/{code}.zip"
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            ["curl", "-sS", "--fail", "--connect-timeout", "30", "--max-time", "900",
             "--speed-limit", "10000", "--speed-time", "120", url, "-o", str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return True
        err = proc.stderr.strip()[:160]
        if attempt < attempts:
            wait = 10 * attempt
            print(f"    attempt {attempt}/{attempts} failed ({err}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
        else:
            print(f"    download failed after {attempts} attempts: {err}", flush=True)
    dest.unlink(missing_ok=True)
    return False


def process_archive(code: str, zip_path: Path, writer, cfg: dict, proc_dir: Path
                    ) -> tuple[int, int]:
    """Render every frame in one archive into both variants. Returns (written, skipped)."""
    label = CLASS_OF[code]
    written = skipped = 0
    depth = cfg["depth"]

    with zipfile.ZipFile(zip_path) as z:
        frames = sorted(
            n for n in z.namelist()
            if "images/" in n and n.endswith(".tif") and n.count("/") == depth
        )
        if not frames:
            raise ValueError(
                f"no frames matched at depth {depth} in {code}. Archive layout "
                f"changed? First few entries: {z.namelist()[:3]}"
            )
        for name in frames:
            # midv500: 19_esp_drvlic/images/TA/TA19_02.tif   -> clip at index 2
            # midv2019:               images/LG/LG01_19.tif  -> clip at index 1
            clip = name.split("/")[depth - 1]
            stem = Path(name).stem                    # "TA19_02"
            condition = cfg["conditions"].get(clip[0], "unknown")
            device = cfg["devices"].get(clip[1], "unknown")

            # Replace the first path segment only. MIDV-500 paths start with the
            # document code ("19_esp_drvlic/images/..."), MIDV-2019 paths start at
            # "images/" -- so matching on a leading slash silently fails on one of
            # them and every crop comes out empty.
            gt_name = name.replace("images/", "ground_truth/", 1).replace(".tif", ".json")
            try:
                quad = json.loads(z.read(gt_name))["quad"]
            except (KeyError, json.JSONDecodeError):
                quad = None

            try:
                img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            except Exception as exc:                  # noqa: BLE001 - log and continue
                print(f"    unreadable frame {name}: {exc}", flush=True)
                skipped += 1
                continue

            rel = Path(label) / code / clip / f"{stem}.jpg"

            full_path = proc_dir / "full" / rel
            full_path.parent.mkdir(parents=True, exist_ok=True)
            letterbox(img).save(full_path, "JPEG", quality=90)

            crop_ok = False
            if quad is not None:
                cropped = rectify(img, quad)
                if cropped is not None:
                    crop_path = proc_dir / "crop" / rel
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cropped.save(crop_path, "JPEG", quality=90)
                    crop_ok = True

            writer.writerow({
                "path": str(rel).replace("\\", "/"),
                "doc_type": code,
                "label": label,
                "clip": clip,
                "condition": condition,
                "device": device,
                "frame": stem,
                "has_crop": int(crop_ok),
            })
            written += 1
            if not crop_ok:
                skipped += 1

    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="midv500")
    ap.add_argument("--limit", type=int, help="only the first N document types")
    ap.add_argument("--only", nargs="*", help="document type numbers, e.g. 19 49")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    proc_dir = PROJECT_ROOT / "data" / cfg["proc"]
    manifest_path = proc_dir / "manifest.csv"

    todo = DOC_TYPES
    if args.only:
        wanted = {n.zfill(2) for n in args.only}
        todo = [c for c in todo if c[:2] in wanted]
    if args.limit:
        todo = todo[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: a document type is done if it already has a marker file.
    done_dir = proc_dir / ".done"
    done_dir.mkdir(exist_ok=True)

    fields = ["path", "doc_type", "label", "clip", "condition", "device", "frame", "has_crop"]
    fresh = not manifest_path.exists()
    manifest_file = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=fields)
    if fresh:
        writer.writeheader()

    total_written = total_skipped = 0
    start = time.time()

    for i, code in enumerate(todo, 1):
        marker = done_dir / f"{code}.ok"
        if marker.exists():
            print(f"[{i}/{len(todo)}] {code} -- already done, skipping", flush=True)
            continue

        print(f"[{i}/{len(todo)}] {code} ({CLASS_OF[code]}) -- downloading", flush=True)
        zip_path = RAW_DIR / f"{code}.zip"
        t0 = time.time()
        if not download(code, zip_path, cfg["ftp"]):
            zip_path.unlink(missing_ok=True)
            continue
        dl = time.time() - t0

        try:
            written, skipped = process_archive(code, zip_path, writer, cfg, proc_dir)
        except (zipfile.BadZipFile, ValueError) as exc:
            print(f"    {exc if isinstance(exc, ValueError) else 'corrupt archive'}"
                  f" -- will retry on next run", flush=True)
            zip_path.unlink(missing_ok=True)
            continue
        finally:
            zip_path.unlink(missing_ok=True)

        manifest_file.flush()
        marker.touch()
        total_written += written
        total_skipped += skipped
        print(
            f"    {written} frames ({skipped} without usable crop) "
            f"| dl {dl:.0f}s, total {time.time() - t0:.0f}s",
            flush=True,
        )

    manifest_file.close()
    shutil.rmtree(RAW_DIR, ignore_errors=True)

    mins = (time.time() - start) / 60
    print(f"\nDone: {total_written} frames written in {mins:.1f} min", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
