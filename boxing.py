# =====================================================================
# make_boxes.py — auto-generate YOLO bounding boxes with MegaDetector
# =====================================================================
# Replaces the full-frame placeholder labels with REAL tight boxes around
# each animal, so YOLO learns to localize (and therefore to detect multiple
# animals per photo). No hand-labeling.
#
# MegaDetector finds animals (not species). Your coarse class (bird/mammal)
# comes from the manifest — we already know what species each image was
# downloaded for, so every box in image X gets X's coarse class. Species
# is the classifier's job downstream.
#
# Install (PyTorch MegaDetector v6 via the 'PytorchWildlife' package):
#   pip install PytorchWildlife
# (CPU works but is slow; a GPU makes this ~10-30x faster.)
#
#   python make_boxes.py
#
# Reads:  dataset/manifest.json, dataset/images/{train,val}/*.jpg
# Writes: dataset/labels/{train,val}/*.txt   (overwrites placeholders)

import json
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans

OUT = Path("dataset")
DETECTOR_CLASSES = ["bird", "mammal"]
CONF = 0.20                 # keep boxes above this MegaDetector confidence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# map each image stem -> coarse class index, from the download manifest
manifest = json.loads((OUT / "manifest.json").read_text())
stem_to_cls = {m["stem"]: DETECTOR_CLASSES.index(m["coarse"]) for m in manifest}

# MegaDetector v6 (MDv6). 'animal' is category 0 in its output; we only keep animals.
model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True)

def yolo_lines_for(img_path, cls_id):
    """Run MegaDetector on one image, return YOLO label lines for animal boxes."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    results = model.single_image_detection(img)
    # results['detections'] holds xyxy boxes, labels, scores (PytorchWildlife API)
    dets = results.get("detections")
    lines = []
    if dets is None:
        return lines
    for box, label, score in zip(dets.xyxy, dets.class_id, dets.confidence):
        if score < CONF:
            continue
        # MegaDetector class 0 == animal (1 person, 2 vehicle) — keep only animals
        if int(label) != 0:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

stats = Counter()
for split in ["train", "val"]:
    imgs = sorted((OUT / "images" / split).glob("*.jpg"))
    for p in tqdm(imgs, desc=f"boxing {split}"):
        cls_id = stem_to_cls.get(p.stem)
        if cls_id is None:
            continue
        lines = yolo_lines_for(p, cls_id)
        label_path = OUT / "labels" / split / f"{p.stem}.txt"
        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            stats["with_box"] += 1
            stats["boxes"] += len(lines)
        else:
            # no animal found by MegaDetector — drop the image from training
            # (a label-less image confuses YOLO; better to exclude it)
            label_path.write_text("")   # empty = background image, or:
            stats["no_detection"] += 1

print("\n================ BOXING SUMMARY ================")
print(f"images with >=1 box : {stats['with_box']}")
print(f"total boxes drawn   : {stats['boxes']}")
print(f"no animal detected  : {stats['no_detection']}  "
      f"(empty labels — review or delete these)")
avg = stats["boxes"] / max(stats["with_box"], 1)
print(f"avg boxes/image     : {avg:.2f}")
print("\nNote: avg ~1.0 is expected (GBIF photos are mostly single-subject).")
print("That's fine — YOLO still learns to localize, so it detects MULTIPLE")
print("animals at inference. Tight boxes are what matter, not box count.")
print("\nNext: train (cell2_train.py). The placeholders are now real boxes.")