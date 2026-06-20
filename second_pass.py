# =====================================================================
# second_pass.py — recover MegaDetector misses with a general COCO YOLO.
# =====================================================================
# MegaDetector (make_boxes.py) misses close-up / frame-filling photos
# because it's trained on distant camera-trap framing. A general COCO-
# trained YOLO is trained on exactly that kind of photo, so it recovers
# many of them — especially BIRDS (COCO has a 'bird' class).
#
# SAFETY: this reads dataset/misses.json and runs ONLY on those images.
# It NEVER touches images that already have boxes. It writes a label only
# when YOLO finds an animal class (your rule: animal/bird -> label,
# anything else -> no box). Recovered images are removed from misses.
#
# COCO mammal coverage is thin (cat/dog/horse/sheep/cow/bear/elephant/
# zebra/giraffe) — your shrews, voles, lynx, beaver etc. are NOT COCO
# classes, so most mammal misses won't be recovered here and will still
# need the hand tool. That's expected.
#
#   pip install ultralytics
#   python second_pass.py            # uses yolo11x (most accurate)
#
# Reads:  dataset/misses.json
# Writes: dataset/labels/<split>/<stem>.txt  (only for newly-found animals)
#         dataset/misses.json  (rewritten — recovered images removed)
#         dataset/second_pass_report.txt

import json
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

OUT = Path("dataset")
DETECTOR_CLASSES = ["bird", "mammal"]
CONF = 0.25
MODEL = "yolo11x.pt"          # large COCO model; only runs on the misses

# COCO class name -> our coarse class. Everything else -> no box.
COCO_TO_COARSE = {
    "bird": "bird",
    "cat": "mammal", "dog": "mammal", "horse": "mammal", "sheep": "mammal",
    "cow": "mammal", "bear": "mammal", "elephant": "mammal",
    "zebra": "mammal", "giraffe": "mammal",
}

misses = json.loads((OUT / "misses.json").read_text())
if not misses:
    print("misses.json is empty — nothing to do.")
    raise SystemExit

print(f"loading {MODEL} (COCO)…")
model = YOLO(MODEL)
names = model.names   # id -> coco name

stats = Counter()
still_missing = []

for m in misses:
    p = Path(m["path"])
    if not p.exists():
        continue
    # what coarse class was this image downloaded as? (used to keep only
    # boxes consistent with the known species; a bird photo should yield a
    # bird box, not a stray 'dog' detection)
    want = m["coarse"]
    res = model.predict(str(p), conf=CONF, imgsz=640, verbose=False)[0]
    W, H = res.orig_shape[1], res.orig_shape[0]

    lines = []
    for b in res.boxes:
        coco = names[int(b.cls)]
        coarse = COCO_TO_COARSE.get(coco)
        if coarse is None:
            continue                      # not an animal class -> your rule: no box
        if coarse != want:
            continue                      # ignore detections that disagree with the known label
        cls_id = DETECTOR_CLASSES.index(coarse)
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        if w > 0 and h > 0:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if lines:
        # write the box — NEVER overwrites an existing box, because by
        # definition every image in misses.json had no box.
        (OUT / "labels" / m["split"] / f"{m['stem']}.txt").write_text(
            "\n".join(lines) + "\n")
        stats["recovered"] += 1
        stats[f"recovered_{want}"] += 1
    else:
        still_missing.append(m)
        stats["still_missing"] += 1

# rewrite misses.json with only the ones still unsolved
(OUT / "misses.json").write_text(json.dumps(still_missing, indent=2, ensure_ascii=False))

report = [
    "================ SECOND PASS (COCO YOLO) ================",
    f"misses processed   : {len(misses)}",
    f"recovered          : {stats['recovered']}  "
    f"(birds {stats['recovered_bird']}, mammals {stats['recovered_mammal']})",
    f"still missing      : {stats['still_missing']}  -> hand-label these",
    "",
    "Remaining misses are mostly mammals COCO doesn't know (shrews, voles,",
    "lynx, etc.) — hand-label in label_tool.html, or full-frame fallback.",
]
report = "\n".join(report)
(OUT / "second_pass_report.txt").write_text(report)
print("\n" + report)