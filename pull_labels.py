"""
pull_labels.py — pull hand-drawn labels from Supabase down to your dataset,
and move each newly-labeled image OUT of _unused/ back into its split so it
rejoins training.

Run on the machine that has the dataset (your dev box, then sync to LUMI):

  pip install supabase
  export SUPABASE_URL="https://.supabase.co"
  export SUPABASE_KEY="<secret key sb_secret_...>"
  python pull_labels.py

For each label row in Supabase:
  - writes dataset/labels/<split>/<stem>.txt
  - moves dataset/images/_unused/<stem>.jpg -> dataset/images/<split>/<stem>.jpg
Idempotent. Re-run anytime to grab the latest labels your gf drew.
"""
import os, shutil
from pathlib import Path
from supabase import create_client

OUT = Path("dataset")
UNUSED = OUT / "images" / "_unused"

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
sb = create_client(url, key)

rows = sb.table("labels").select("*").execute().data
print(f"{len(rows)} labels in Supabase")

wrote = 0
moved = 0
for r in rows:
    stem, split, lines = r["stem"], r["split"], r["lines"]
    if not lines:
        continue
    lp = OUT / "labels" / split / f"{stem}.txt"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("\n".join(lines) + "\n")
    wrote += 1
    src = UNUSED / f"{stem}.jpg"
    dst = OUT / "images" / split / f"{stem}.jpg"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

print(f"wrote {wrote} label files; moved {moved} images back into training")
print("now sync dataset/ to LUMI and retrain.")
print("tip: rsync -av dataset/ lumi:/path/to/birds/dataset/")