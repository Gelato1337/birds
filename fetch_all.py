# =====================================================================
# fetch_all.py — PARALLEL, RESUMABLE background download of the full
#                Finnish bird+mammal dataset (~50 GB target).
# =====================================================================
# Fixes the slow sequential pull: concurrent image downloads + concurrent
# GBIF page fetches, a progress ledger so you can stop/resume any time, and
# per-species caps so the long tail stays balanced.
#
#   python build_species_list.py        # first: makes species.json
#   nohup python fetch_all.py > fetch.log 2>&1 &   # run in background
#   tail -f fetch.log                    # watch progress
#
# Re-run any time; it skips species/images already done (idempotent).

import io, json, time, os, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from pygbif import occurrences as occ

# ---------------- config ----------------
OUT = Path("dataset")
SPECIES_JSON = "species.json"      # from build_species_list.py
PER_SPECIES = 350                  # cap per species (balance long tail)
IMG_MAX = 640
VAL_EVERY = 8
DL_WORKERS = 32                    # concurrent image downloads (the speed win)
PAGE_WORKERS = 4                   # concurrent GBIF search pages
DETECTOR_CLASSES = ["bird", "mammal"]
TIMEOUT = 15

LEDGER = OUT / "_ledger.json"      # tracks completed species + global index
LOCK = threading.Lock()

for s in ["train", "val"]:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
(OUT / "crops").mkdir(parents=True, exist_ok=True)

# ---------------- ledger (resume support) ----------------
if LEDGER.exists():
    ledger = json.loads(LEDGER.read_text())
else:
    ledger = {"done_species": [], "idx": 0, "manifest": []}

def save_ledger():
    with LOCK:
        LEDGER.write_text(json.dumps(ledger))

# ---------------- GBIF media URLs (paged, concurrent) ----------------
def media_urls(taxon_key, limit):
    """Collect up to `limit` (url, lat, lon) for a species, paging concurrently."""
    # first call to learn count
    head = occ.search(taxonKey=taxon_key, country="FI", mediaType="StillImage",
                      hasCoordinate=True, limit=1)
    total = min(limit, head.get("count", 0))
    offsets = list(range(0, total, 300))
    out = []

    def fetch_page(off):
        r = occ.search(taxonKey=taxon_key, country="FI", mediaType="StillImage",
                       hasCoordinate=True, limit=300, offset=off)
        page = []
        for rec in r.get("results", []):
            url = next((m.get("identifier") for m in rec.get("media", [])
                        if m.get("type") == "StillImage"), None)
            if url:
                page.append((url, rec.get("decimalLatitude"),
                             rec.get("decimalLongitude")))
        return page

    with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as ex:
        for pg in ex.map(fetch_page, offsets):
            out.extend(pg)
    return out[:limit]

# ---------------- image download ----------------
def grab_and_save(url, lat, lon, species, coarse, cls):
    try:
        b = requests.get(url, timeout=TIMEOUT)
        b.raise_for_status()
        im = Image.open(io.BytesIO(b.content)).convert("RGB")
        im.thumbnail((IMG_MAX, IMG_MAX))
    except Exception:
        return None
    with LOCK:
        idx = ledger["idx"]; ledger["idx"] += 1
    split = "val" if idx % VAL_EVERY == 0 else "train"
    stem = f"{idx:08d}"
    im.save(OUT / "images" / split / f"{stem}.jpg", quality=88)
    (OUT / "labels" / split / f"{stem}.txt").write_text(f"{cls} 0.5 0.5 1.0 1.0\n")
    return dict(stem=stem, split=split, species=species, coarse=coarse,
               cls=cls, lat=lat, lon=lon)

# ---------------- main loop ----------------
def main():
    species = json.loads(Path(SPECIES_JSON).read_text())
    todo = [s for s in species if s["taxonKey"] not in ledger["done_species"]]
    print(f"{len(species)} species total; {len(todo)} remaining to fetch")

    for si, sp in enumerate(todo):
        key = sp["taxonKey"]; coarse = sp["coarse"]
        cls = DETECTOR_CLASSES.index(coarse)
        t0 = time.time()
        urls = media_urls(key, PER_SPECIES)
        if not urls:
            ledger["done_species"].append(key); save_ledger(); continue

        got = 0
        with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
            futs = [ex.submit(grab_and_save, u, la, lo, sp["name"], coarse, cls)
                    for (u, la, lo) in urls]
            for f in as_completed(futs):
                r = f.result()
                if r:
                    with LOCK: ledger["manifest"].append(r)
                    got += 1

        ledger["done_species"].append(key)
        save_ledger()
        dt = time.time() - t0
        print(f"[{si+1}/{len(todo)}] {sp['name']:<28} {got:>4} imgs "
              f"in {dt:4.0f}s  (total idx={ledger['idx']})", flush=True)

    # write data.yaml + manifest
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(DETECTOR_CLASSES)}\nnames: {DETECTOR_CLASSES}\n")
    (OUT / "manifest.json").write_text(json.dumps(ledger["manifest"], indent=2,
                                                   ensure_ascii=False))
    # quick summary
    cnt = Counter(m["species"] for m in ledger["manifest"])
    sz = sum(p.stat().st_size for p in OUT.rglob("*.jpg")) / 1e9
    print(f"\n✅ done. {ledger['idx']} images, {len(cnt)} species, ~{sz:.1f} GB on disk")

if __name__ == "__main__":
    main()