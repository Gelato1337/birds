# =====================================================================
# make_boxes.py — auto-generate YOLO bounding boxes with MegaDetector
# =====================================================================
# Stage 1 of labeling: run the auto-labeler, REPORT the hit rate, and write
# a list of images it couldn't box (misses.json) so you can hand-label only
# those in the companion tool (label_tool.html).
#
# MegaDetector finds animals (not species). The coarse class (bird/mammal)
# for each box comes from the manifest — we already know what species each
# image was downloaded for. Species ID is the classifier's job downstream.
#
# Install (PyTorch MegaDetector v6):
#   pip install PytorchWildlife
# GPU strongly recommended (10-30x faster than CPU).
#
#   python make_boxes.py
#
# Reads:  dataset/manifest.json, dataset/images/{train,val}/*.jpg
# Writes: dataset/labels/{train,val}/*.txt   (real boxes for hits)
#         dataset/misses.json                (images with no detection)
#         dataset/box_report.txt             (summary + per-species hit rate)

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from PytorchWildlife.models import detection as pw_detection

OUT = Path("dataset")
DETECTOR_CLASSES = ["bird", "mammal"]
CONF = 0.20                 # keep boxes above this MegaDetector confidence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

manifest = json.loads((OUT / "manifest.json").read_text())
stem_to_meta = {m["stem"]: m for m in manifest}

print(f"loading MegaDetector v6 on {DEVICE}...")
# Newer PytorchWildlife requires an explicit model version. Options:
#   MDV6-yolov9-c (fast, good default), MDV6-yolov9-e (more accurate, slower),
#   MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c
MD_VERSION = "MDV6-yolov9-c"
model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version=MD_VERSION)


def boxes_for(img_path, cls_id):
    """Run MegaDetector on one image, return YOLO label lines for animal boxes.
    API note: this PytorchWildlife version takes a numpy RGB array and returns
    {'detections': sv.Detections, 'labels': [str,...]}. We keep only boxes whose
    label starts with 'animal' (MegaDetector also emits 'person'/'vehicle')."""
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    arr = np.array(pil)
    res = model.single_image_detection(arr)
    dets = res.get("detections")
    labels = res.get("labels", [])
    lines = []
    if dets is None or len(dets.xyxy) == 0:
        return lines
    confs = getattr(dets, "confidence", None)
    for i, box in enumerate(dets.xyxy):
        # label string looks like 'animal 0.93'; confidence also in dets.confidence
        lab = labels[i] if i < len(labels) else ""
        conf = float(confs[i]) if confs is not None else 1.0
        if conf < CONF:
            continue
        if not str(lab).lower().startswith("animal"):
            continue   # drop person / vehicle
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


stats = Counter()
per_species = defaultdict(lambda: [0, 0])   # species -> [hits, total]
misses = []

for split in ["train", "val"]:
    imgs = sorted((OUT / "images" / split).glob("*.jpg"))
    for p in tqdm(imgs, desc=f"boxing {split}"):
        meta = stem_to_meta.get(p.stem)
        if not meta:
            continue
        cls_id = DETECTOR_CLASSES.index(meta["coarse"])
        lines = boxes_for(p, cls_id)
        label_path = OUT / "labels" / split / f"{p.stem}.txt"
        per_species[meta["species"]][1] += 1
        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            stats["hit"] += 1
            stats["boxes"] += len(lines)
            per_species[meta["species"]][0] += 1
        else:
            # leave NO label file yet — these go to the hand-labeling tool.
            stats["miss"] += 1
            misses.append({"stem": p.stem, "split": split,
                           "path": str(p), "species": meta["species"],
                           "coarse": meta["coarse"]})

(OUT / "misses.json").write_text(json.dumps(misses, indent=2, ensure_ascii=False))

total = stats["hit"] + stats["miss"]
lines = [
    "================ AUTO-LABEL REPORT ================",
    f"images processed : {total}",
    f"auto-boxed (hit) : {stats['hit']}  ({100*stats['hit']//max(total,1)}%)",
    f"no detection     : {stats['miss']}  -> dataset/misses.json (hand-label these)",
    f"total boxes      : {stats['boxes']}",
    f"avg boxes/hit    : {stats['boxes']/max(stats['hit'],1):.2f}",
    "",
    "Per-species hit rate (worst first — these need the most hand-labeling):",
]
worst = sorted(per_species.items(), key=lambda kv: kv[1][0]/max(kv[1][1], 1))
for sp, (hit, tot) in worst:
    rate = 100 * hit // max(tot, 1)
    lines.append(f"  {rate:3d}%  {hit:4d}/{tot:<4d}  {sp}")

report = "\n".join(lines)
(OUT / "box_report.txt").write_text(report)
print("\n" + report)
print(f"\nNext: open label_tool.html and load dataset/misses.json to hand-label "
      f"the {stats['miss']} misses. Then run merge_labels.py.")