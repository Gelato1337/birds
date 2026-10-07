"""
verify_conflicts.py — find_conflicts() groups by a 64-bit perceptual hash, which
is fast but COLLIDES: different photos can share a hash (e.g. 'beaver vs jaeger').
This confirms each candidate group by actually COMPARING PIXELS, so only genuine
same-photo / different-species conflicts survive.

Rule: keep a group only if ALL its member images are near-identical (mean abs
grayscale diff < THRESH). Because find_conflicts only forms a group when 2+
DIFFERENT species are present, "all images identical" == one real photo tagged
as several species == a true conflict. Groups that split into different pictures
are hash collisions and get dropped.

Writes conflicts_verified.json (what populate_labeler.py / pull_conflicts.py read)
and rebuilds conflicts_review/ with descriptive filenames so you can eyeball them
before uploading anything.

  python verify_conflicts.py
"""
import json, shutil, re
from pathlib import Path
from PIL import Image
import numpy as np

OUT = Path("dataset_new")
CONF = OUT / "conflict_manifest.json"
REVIEW = Path("conflicts_review")
VERIFIED = Path("conflicts_verified.json")
THRESH = 10.0          # mean abs grayscale diff (0-255); below = same photo
MAX_COPIES = 8

def degenerate(h):
    return len(set(h)) <= 3

def small(stem):
    p = OUT / "conflicts" / f"{stem}.jpg"
    if not p.exists(): return None
    try:
        with Image.open(p) as im:
            return np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
    except Exception:
        return None

def san(s):
    return re.sub(r"[^A-Za-z0-9-]+", "-", s).strip("-")

groups = [g for g in json.loads(CONF.read_text())
          if not degenerate(g["img_hash"]) and len(g["stems"]) <= MAX_COPIES]
print(f"{len(groups)} candidate groups to pixel-verify…")

real = []
for g in groups:
    arrs = [(st, small(st)) for st in g["stems"]]
    arrs = [(st, a) for st, a in arrs if a is not None]
    if len(arrs) < 2:
        continue
    base = arrs[0][1]
    if all(np.mean(np.abs(a - base)) < THRESH for _, a in arrs[1:]):
        real.append({**g, "stems": [st for st, _ in arrs]})   # genuine same-photo conflict

VERIFIED.write_text(json.dumps(real, ensure_ascii=False))
print(f"VERIFIED real conflicts: {len(real)} (of {len(groups)} candidates)")

if REVIEW.exists(): shutil.rmtree(REVIEW)
REVIEW.mkdir()
for i, g in enumerate(sorted(real, key=lambda g: tuple(g["species"]))):
    rep = g["stems"][0]; src = OUT / "conflicts" / f"{rep}.jpg"
    if not src.exists(): continue
    name = f"{i:03d}__{'__VS__'.join(san(s) for s in g['species'])}__{rep}.jpg"
    shutil.copy(src, REVIEW / name)
print(f"review folder rebuilt: {REVIEW.resolve()}  ({len(real)} images)")
