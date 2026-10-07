"""
pull_conflicts.py — apply the conflict resolutions (from the labeler -> Supabase)
back into dataset_new. For each cross-species duplicate the user adjudicated:

  - species chosen  -> re-box the photo with YOLO, add it to the manifest as that
                       species, move it into images/{train,val}; delete the other
                       (identical) copies of the group.
  - neither / junk  -> discard every copy (unidentifiable / not either species).
  - unresolved      -> leave in conflicts/ for a later pass.

Run on the box with the dataset + YOLO:
  pip install supabase ultralytics
  export SUPABASE_URL="https://ljuendcuoruxorjdcdmu.supabase.co"
  export SUPABASE_KEY="<secret key sb_secret_...>"
  python pull_conflicts.py
Idempotent — re-run anytime to apply the latest resolutions.
"""
import os, json
from pathlib import Path
from supabase import create_client
from ultralytics import YOLO

OUT = Path("dataset_new")
VERIFIED = Path("conflicts_verified.json")   # the pixel-verified conflict groups
MODEL = "yolo11x.pt"
CONF_TH = 0.30
MAX_BOX_FRAC = 0.985
VAL_EVERY = 8
DETECTOR_CLASSES = ["bird", "mammal"]
COCO_TO_COARSE = {
    "bird": "bird",
    "cat": "mammal", "dog": "mammal", "horse": "mammal", "sheep": "mammal",
    "cow": "mammal", "bear": "mammal", "elephant": "mammal",
    "zebra": "mammal", "giraffe": "mammal",
}

species_meta = {s["name"]: s for s in json.loads(Path("species.json").read_text())}

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
sb = create_client(url, key)
rows = {r["stem"]: r for r in
        sb.table("label_items").select("stem,label").eq("status", "done").execute().data}
print(f"{len(rows)} resolved (done) items in Supabase")

groups = json.loads(VERIFIED.read_text())
manifest = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else []

det = None   # YOLO is only needed for the legacy single-species fallback; load lazily

def box_image(path, want_coarse):
    global det
    if det is None:
        det = YOLO(MODEL)
    NAMES = det.names
    res = det.predict(str(path), conf=CONF_TH, imgsz=640, verbose=False)[0]
    W, H = res.orig_shape[1], res.orig_shape[0]
    boxes = []
    for b in res.boxes:
        co = COCO_TO_COARSE.get(NAMES[int(b.cls)])
        if co != want_coarse: continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        if w <= 0 or h <= 0: continue
        if w >= MAX_BOX_FRAC and h >= MAX_BOX_FRAC: continue
        boxes.append((DETECTOR_CLASSES.index(co), cx, cy, w, h))
    return boxes

resolved = discarded = noboxed = unresolved = 0
remaining, multi_hold = [], []
for g in groups:
    rep = g["stems"][0]
    row = rows.get(rep)
    if not row:
        remaining.append(g); unresolved += 1; continue
    lab = row.get("label") or {}
    paths = [OUT / "conflicts" / f"{st}.jpg" for st in g["stems"]]
    src = OUT / "conflicts" / f"{rep}.jpg"
    if lab.get("noise"):                          # not a species / junk -> drop all copies
        for p in paths: p.unlink(missing_ok=True)
        discarded += 1; continue

    # ---- hand-drawn per-box species (current labeler) ----
    if lab.get("boxes"):
        lines, recs = [], []
        for b in lab["boxes"]:
            meta = species_meta.get(b.get("species"))
            if not meta:
                continue                          # box tagged with a non-313 species -> skip box
            cls = DETECTOR_CLASSES.index(meta["coarse"])
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2   # labeler stores top-left + wh
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {b['w']:.6f} {b['h']:.6f}")
            recs.append({"stem": rep, "species": b["species"], "coarse": meta["coarse"], "cls": cls,
                         "lat": None, "lon": None, "source_url": g["source_urls"][0],
                         "source": "conflict-resolved", "img_hash": g["img_hash"],
                         "box": {"cx": round(cx, 6), "cy": round(cy, 6), "w": round(b["w"], 6), "h": round(b["h"], 6)}})
        if not lines:
            remaining.append(g); unresolved += 1; continue
        split = "val" if int(rep) % VAL_EVERY == 0 else "train"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        if src.exists(): src.rename(OUT / "images" / split / f"{rep}.jpg")
        (OUT / "labels" / split / f"{rep}.txt").write_text("".join(l + "\n" for l in lines))
        for r in recs: r["split"] = split; r["n_boxes"] = len(lines); manifest.append(r)
        for st in g["stems"][1:]: (OUT / "conflicts" / f"{st}.jpg").unlink(missing_ok=True)
        resolved += 1; continue

    # ---- legacy single-species pick (no boxes) -> YOLO re-box ----
    sps = [s for s in (lab.get("species") or []) if s]
    if not sps:
        remaining.append(g); unresolved += 1; continue
    if len(sps) > 1:
        multi_hold.append({**g, "species_chosen": sps}); continue
    meta = species_meta.get(sps[0])
    if not meta:
        remaining.append(g); unresolved += 1; continue
    coarse = meta["coarse"]; cls = DETECTOR_CLASSES.index(coarse)
    boxes = box_image(src, coarse) if src.exists() else []
    if not boxes:
        remaining.append(g); noboxed += 1; continue
    split = "val" if int(rep) % VAL_EVERY == 0 else "train"
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    src.rename(OUT / "images" / split / f"{rep}.jpg")
    (OUT / "labels" / split / f"{rep}.txt").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for (c, cx, cy, w, h) in boxes))
    manifest.append({"stem": rep, "species": sps[0], "coarse": coarse, "cls": cls,
                     "lat": None, "lon": None, "source_url": g["source_urls"][0],
                     "source": "conflict-resolved", "img_hash": g["img_hash"],
                     "split": split, "n_boxes": len(boxes)})
    for st in g["stems"][1:]:
        (OUT / "conflicts" / f"{st}.jpg").unlink(missing_ok=True)
    resolved += 1

(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
VERIFIED.write_text(json.dumps(remaining, ensure_ascii=False))
if multi_hold:
    Path("multi_resolved.json").write_text(json.dumps(multi_hold, ensure_ascii=False))
print(f"resolved {resolved} (added to training), discarded {discarded}, "
      f"no-box {noboxed} (left for manual), unresolved {unresolved}, "
      f"multi-species {len(multi_hold)} (-> multi_resolved.json, need box-level)")
print(f"manifest now {len(manifest)}")
