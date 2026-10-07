"""
populate_labeler.py — seed the Supabase labeling queue with the verified
conflicts. Uploads each conflict's representative image to Storage and inserts a
row into the label_items table (status 'pending', with the candidate species).
NO json index — the labeler reads the queue straight from the DB.

Run AFTER:
  1. you've cleared the old bucket (python delete_misses.py --yes), and
  2. run the label_items block from schema.sql in the Supabase SQL editor.

  pip install supabase
  export SUPABASE_URL="https://ljuendcuoruxorjdcdmu.supabase.co"
  export SUPABASE_KEY="<your SECRET key (sb_secret_...)>"
  python populate_labeler.py

Idempotent: re-running re-uploads missing images and upserts the rows.
"""
import os, json
from pathlib import Path
from supabase import create_client

BUCKET = "misses"                          # reusing the existing public bucket
OUT = Path("dataset_new")
VERIFIED = Path("conflicts_verified.json")  # from verify_conflicts.py

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]            # SECRET key for upload + insert
sb = create_client(url, key)

if not VERIFIED.exists():
    raise SystemExit("run verify_conflicts.py first (writes conflicts_verified.json)")
groups = json.loads(VERIFIED.read_text())
pub = f"{url}/storage/v1/object/public/{BUCKET}/"

try:
    existing = {o["name"] for o in sb.storage.from_(BUCKET).list()}
except Exception:
    existing = set()

rows, uploaded = [], 0
for g in groups:
    stem = g["stems"][0]                    # one representative; all copies identical
    p = OUT / "conflicts" / f"{stem}.jpg"
    if not p.exists():
        continue
    name = f"{stem}.jpg"
    if name not in existing:
        try:
            sb.storage.from_(BUCKET).upload(
                name, p.read_bytes(),
                {"content-type": "image/jpeg", "upsert": "true"})
            uploaded += 1
            if uploaded % 50 == 0:
                print(f"  uploaded {uploaded}…")
        except Exception as e:
            print(f"  upload failed {name}: {e}")
            continue
    rows.append({"stem": stem, "url": pub + name,
                 "candidates": g["species"], "status": "pending"})

inserted = 0
for i in range(0, len(rows), 100):
    sb.table("label_items").upsert(rows[i:i + 100], on_conflict="stem").execute()
    inserted += len(rows[i:i + 100])

print(f"\n✅ uploaded {uploaded} images, queued {inserted} items into label_items.")
print("open the app -> You -> Merkintä (or labeler.html), sign in, and start labeling.")
