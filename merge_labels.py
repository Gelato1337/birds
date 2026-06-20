# =====================================================================
# merge_labels.py — fold hand-labeled boxes back into the dataset.
# =====================================================================
# After make_boxes.py (auto-labels the easy ones) and label_tool.html
# (hand-label the misses -> hand_labels.json), this writes the hand boxes
# into dataset/labels/, and removes any images that ended up with NO box
# at all (neither auto nor hand) from the train/val sets, since a YOLO
# image with no label hurts training.
#
#   python merge_labels.py
#
# Reads:  dataset/misses.json, hand_labels.json, dataset/labels/**
# Writes: dataset/labels/{split}/<stem>.txt for hand-labeled images
#         moves boxless images to dataset/images/_unused/ (not deleted)

import json, shutil
from pathlib import Path

OUT = Path("dataset")
hand = json.loads(Path("hand_labels.json").read_text()) if Path("hand_labels.json").exists() else {}
misses = json.loads((OUT / "misses.json").read_text())

# 1. write hand-labeled boxes
written = 0
for stem, v in hand.items():
    lp = OUT / "labels" / v["split"] / f"{stem}.txt"
    lp.write_text("\n".join(v["lines"]) + "\n")
    written += 1

# 2. any miss NOT hand-labeled => no box at all => move image out of training
unused = OUT / "images" / "_unused"
unused.mkdir(exist_ok=True)
moved = 0
for m in misses:
    if m["stem"] in hand:
        continue
    src = Path(m["path"])
    if src.exists():
        shutil.move(str(src), str(unused / src.name))
        # remove any stray empty label
        lp = OUT / "labels" / m["split"] / f"{m['stem']}.txt"
        if lp.exists(): lp.unlink()
        moved += 1

print("================ MERGE COMPLETE ================")
print(f"hand-labeled boxes written : {written}")
print(f"boxless images moved aside : {moved}  -> dataset/images/_unused/")
print(f"(moved, not deleted — restore from _unused if you change your mind)")
print("\nDataset is now fully labeled. Run cell2_train.py to train.")