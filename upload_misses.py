"""
upload_misses.py — upload the unlabeled images to Supabase Storage so the
labeler (and your gf) can reach them from anywhere.

Setup once in Supabase dashboard:
  Storage -> New bucket -> name it "misses" -> make it PUBLIC (read).
  (public read is fine; these are just animal photos. writes still need auth.)

Then:
  pip install supabase
  export SUPABASE_URL="https://.supabase.co"
  export SUPABASE_KEY="<your SECRET key (sb_secret_...)>"   # server-side only!
  python upload_misses.py

Uploads dataset/images/_unused/*.jpg to the bucket. Idempotent: skips files
already there. Writes misses_index.json (the list the labeler loads).
"""
import os, json
from pathlib import Path
from supabase import create_client

BUCKET = "misses"
SRC = Path("dataset/images/_unused")

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]          # use the SECRET key for bulk upload
sb = create_client(url, key)

# list what's already in the bucket so re-runs skip them
try:
    existing = {o["name"] for o in sb.storage.from_(BUCKET).list()}
except Exception:
    existing = set()

imgs = sorted(SRC.glob("*.jpg"))
print(f"{len(imgs)} images to upload ({len(existing)} already there)")

index = []
done = 0
for p in imgs:
    name = p.name
    index.append({"stem": p.stem, "file": name})
    if name in existing:
        continue
    try:
        sb.storage.from_(BUCKET).upload(
            name, p.read_bytes(),
            {"content-type": "image/jpeg", "upsert": "true"})
        done += 1
        if done % 200 == 0:
            print(f"  uploaded {done}…")
    except Exception as e:
        print(f"  failed {name}: {e}")

# public URL base for the labeler to build image links
pub = f"{url}/storage/v1/object/public/{BUCKET}/"
Path("misses_index.json").write_text(json.dumps(
    {"base_url": pub, "images": index}, indent=2))

print(f"\n✅ uploaded {done} new ({len(imgs)} total in index).")
print(f"public base: {pub}")
print("wrote misses_index.json — host it alongside labeler.html "
      "(or commit it; it's tiny).")