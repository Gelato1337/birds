# =====================================================================
# fetch_v2.py — two-phase: (1) download fast, (2) box on GPU in batches.
# =====================================================================
# Why two phases: boxing inline while many download threads share one GPU
# serializes on inference and stalls. Separating them = downloads run
# network-parallel, then ONE clean batched GPU pass does all the boxing.
#
# Phase 1 (download):   pull candidate images for under-represented species
#                       to dataset_new/_incoming/<stem>.jpg  (+ _incoming.json)
# Phase 2 (box):        run YOLO once over _incoming in batches.
#                       1+ animals -> auto-box ALL boxes as the queried species
#                                     (multiple animals in one photo are the same
#                                      species; coarse detector only needs bird/mammal)
#                       no animal  -> move to no_box/ for MANUAL labeling.
#                                     (NOT discarded — YOLO can miss a real bird;
#                                      a human decides keep-and-box vs reject.)
#                                     Only truly unreadable images are deleted.
# After boxing:         find_conflicts() flags the same photo (by perceptual hash)
#                       tagged under 2+ species -> conflicts/ for PRIORITY manual
#                       labeling (the real corruption; ~56 such cases last time).
#
#   python fetch_v2.py            # LOOP: download->box->recheck, top up to target
#   python fetch_v2.py download   # only phase 1 (one pass)
#   python fetch_v2.py box        # only phase 2 (one pass)
#
# Target is measured AFTER boxing: TARGET_GOOD (700) boxed images per species.
# Each round downloads ~TARGET_GOOD/BOX_YIELD (~1000) candidates for species below
# target, boxes them, rechecks, and tops up again until target or sources dry up.
#
# SEPARATE FROM THE OLD DATASET. This run writes ONLY to dataset_new/. The
# previous dataset/ (MegaDetector-boxed, known to contain junk: full-frame
# boxes, spiders/bugs labelled as birds) is left untouched. Stem indices
# continue past the old dataset's max so the two can be merged later without
# collisions. Combine happens at the very END, by hand, before LUMI export —
# NOT here. That keeps the new data auditable for purity on its own.
#
# Idempotent + resumable. Re-run any time; this only ADDS to dataset_new/.

import io, json, time, os, sys, math, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from pygbif import occurrences as occ

# ---------------- config ----------------
OLD_OUT = Path("dataset")      # previous dataset — READ-ONLY here (idx offset only)
OUT = Path("dataset_new")      # this run writes ONLY here; combine with OLD later
SPECIES_JSON = "species.json"

# The goal is measured AFTER boxing: TARGET_GOOD boxed single-animal images per
# species. Boxing discards some downloads (no animal / multi / unreadable), so
# we download MORE than the goal to net it: ~TARGET_GOOD / BOX_YIELD attempts
# (700 / 0.7 ~= 1000). After a box pass, any species still under TARGET_GOOD is
# topped up again next round (see run_loop) until it hits target or sources run
# dry.  ->  "download 1000, aim for 700."
TARGET_GOOD = 700              # desired BOXED-good images per species (the real goal)
BOX_YIELD = 0.70              # est. fraction of downloads that survive boxing
MAX_ROUNDS = 4               # download+box top-up rounds per launch (loop mode)
IMG_MAX = 640
VAL_EVERY = 8
DL_WORKERS = 32
PAGE_WORKERS = 4
TIMEOUT = 15
CONF = 0.30
MODEL = "yolo11x.pt"
BATCH = 32
# Descriptive User-Agent — Wikimedia (and good manners elsewhere) require one;
# requests without it get 403'd from commons/upload.wikimedia.org.
USER_AGENT = "FieldmarkBirdFetch/1.0 (Finnish bird+mammal dataset; +https://github.com/Gelato1337/birds)"
HEADERS = {"User-Agent": USER_AGENT}

# ---- sources, in PRIORITY order ----
# We pull from each source in turn until we have TARGET *successful* downloads
# for a species (a download-shortfall cascade — NOT a URL-count cascade, which
# was the bug that left Barnacle Goose at 215: GBIF handed back thousands of
# dead links, the code thought it was full, and never fell through to iNat).
#
# Order is by ID purity, not volume:
#   inat      — research-grade, community-VERIFIED ids, reliable CDN urls. cleanest.
#   gbif      — huge aggregator, but mixed: we exclude museum specimens (dead birds)
#               and many urls are dead, so it's the supplement, not the primary.
#   wikimedia — Commons species category (curated by scientific name), CC/PD, no
#               key. Low volume but clean -> a good top-up for the rare tail.
# (Flickr was dropped: its API needs a Pro account, and its uploader-tag ids
#  were the noisiest source for species accuracy anyway.)
SOURCES = ["inat", "gbif", "wikimedia"]
OVERFETCH = 4                  # fetch this * needed candidate urls per source
                              # (download success rate is low, esp. for gbif)

# Adaptive source priority: if a source produces more than this many no-box
# failures for a species (its photos keep failing the YOLO gate — corrupt class,
# bad framing), DEMOTE it below the others for that species so the next top-up
# tries the cleaner source first.
NOBOX_REPRIORITIZE = 300

# GBIF basis-of-record: keep only real observations of LIVE animals. Excludes
# PRESERVED_SPECIMEN / FOSSIL_SPECIMEN / MATERIAL_SAMPLE (dead museum birds,
# which are exactly the wrong thing to train a live-animal detector on).
GBIF_BASIS = ["HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION"]

# Purity guard: a box covering essentially the whole frame is the MegaDetector
# failure mode (it wrote 0.5 0.5 1.0 1.0 as a fallback, passing junk). YOLO has
# no fallback, but reject near-full-frame detections anyway — they carry no
# localisation signal and are usually a misfire. Both dims must exceed this.
MAX_BOX_FRAC = 0.985

# Transient GBIF backend errors (503 "Backend fetch failed") are common and
# must NOT abort an overnight run. Retry with exponential backoff, then skip.
GBIF_RETRIES = 4

DETECTOR_CLASSES = ["bird", "mammal"]
# This is the COMPLETE set of real-animal classes in COCO-80 — 1 bird + 9
# mammals. There is no generic "animal" class (that was MegaDetector). Do NOT
# add to this map: COCO's "mouse" is a COMPUTER mouse and "teddy bear" is a toy,
# so mapping either would inject junk (desks/plushies) as voles/bears.
# The 9 mammal classes also double as look-alikes for Finnish mammals YOLO
# doesn't know (wolf/fox->dog, deer/reindeer->horse/cow, lynx->cat), which
# gives a correct *box* even when the species is wrong — species ID is the
# downstream classifier's job. Mammals with no COCO look-alike (shrews, voles,
# seals, beaver, hare) get no box -> no_box/ for manual labeling, as expected.
COCO_TO_COARSE = {
    "bird": "bird",
    "cat": "mammal", "dog": "mammal", "horse": "mammal", "sheep": "mammal",
    "cow": "mammal", "bear": "mammal", "elephant": "mammal",
    "zebra": "mammal", "giraffe": "mammal",
}

INCOMING = OUT / "_incoming"
INCOMING_JSON = OUT / "_incoming.json"
LOCK = threading.Lock()
for s in ["train", "val"]:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
(OUT / "no_box").mkdir(parents=True, exist_ok=True)        # YOLO found nothing -> manual
(OUT / "conflicts").mkdir(parents=True, exist_ok=True)     # same photo, 2+ species -> PRIORITY manual
INCOMING.mkdir(parents=True, exist_ok=True)

def next_idx():
    # Continue past BOTH the old dataset and this run's output, so stems are
    # globally unique and the eventual OLD+NEW merge never collides.
    existing = list((OLD_OUT / "images" / "train").glob("*.jpg")) + \
               list((OLD_OUT / "images" / "val").glob("*.jpg")) + \
               list((OUT / "images" / "train").glob("*.jpg")) + \
               list((OUT / "images" / "val").glob("*.jpg")) + \
               list(INCOMING.glob("*.jpg"))
    return max([int(p.stem) for p in existing if p.stem.isdigit()], default=-1) + 1

def current_counts():
    # Count ONLY this run's manifest. The old dataset is suspect and kept
    # separate, so we re-collect every species from scratch up to TARGET.
    mpath = OUT / "manifest.json"
    if not mpath.exists(): return Counter()
    return Counter(m["species"] for m in json.loads(mpath.read_text()))

# ---------------- URL sources ----------------
# NOTE: we do NOT lock to Finland. A species looks the same worldwide, and the
# rare Finnish vagrants (Pied Avocet, Snowy Owl, etc.) have almost no Finnish
# photos but thousands globally. For a visual detector, origin is irrelevant.
COUNTRY = None        # set to "FI" to restrict; None = worldwide (recommended)

def gbif_search(**kw):
    """occ.search with retry/backoff on transient errors. Returns {} on give-up
    so a flaky GBIF backend skips one page/species instead of crashing the run."""
    for attempt in range(GBIF_RETRIES):
        try:
            return occ.search(**kw)
        except Exception as e:
            if attempt == GBIF_RETRIES - 1:
                print(f"  GBIF gave up after {GBIF_RETRIES} tries: {e}", flush=True)
                return {}
            wait = 2 ** attempt
            print(f"  GBIF error ({e}); retry {attempt+1}/{GBIF_RETRIES} in {wait}s", flush=True)
            time.sleep(wait)
    return {}

# Each source pages until it has `limit` UNSEEN urls (skip = urls we already
# have). This is what makes top-up rounds work: a re-fetch skips what we already
# pulled and digs into deeper pages, instead of re-returning the same first page.
def gbif_urls(taxon_key, limit, skip):
    # basisOfRecord excludes preserved/museum specimens -> live animals only.
    kw = dict(taxonKey=taxon_key, mediaType="StillImage", basisOfRecord=GBIF_BASIS)
    if COUNTRY: kw["country"] = COUNTRY
    head = gbif_search(limit=1, **kw)
    total = head.get("count", 0)
    def fetch_page(off):
        r = gbif_search(limit=300, offset=off, **kw)
        pg = []
        for rec in r.get("results", []):
            url = next((m.get("identifier") for m in rec.get("media", [])
                        if m.get("type") == "StillImage"), None)
            if url and url not in skip:
                pg.append((url, rec.get("decimalLatitude"), rec.get("decimalLongitude")))
        return pg
    out, offs = [], list(range(0, total, 300))
    for i in range(0, len(offs), PAGE_WORKERS):
        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as ex:
            for pg in ex.map(fetch_page, offs[i:i+PAGE_WORKERS]):
                out.extend(pg)
        if len(out) >= limit: break          # got enough unseen — stop paging
    return out[:limit]

def inat_urls(sci_name, limit, skip):
    binomial = " ".join(str(sci_name).split()[:2])
    out, page = [], 1
    while len(out) < limit and page <= 50:   # iNat caps page*per_page at 10000
        try:
            r = requests.get("https://api.inaturalist.org/v1/observations",
                params={"taxon_name": binomial, "quality_grade": "research",
                        "photos": "true", "per_page": 200, "page": page,
                        "license": "cc0,cc-by,cc-by-nc"}, timeout=TIMEOUT).json()
        except Exception:
            break
        results = r.get("results", [])
        if not results: break
        for rec in results:
            geo = rec.get("geojson") or {}
            coords = geo.get("coordinates") or [None, None]
            for ph in rec.get("photos", []):
                u = ph.get("url", "").replace("square", "medium")
                if u and u not in skip:
                    out.append((u, coords[1], coords[0]))
        page += 1
    return out[:limit]

def wikimedia_urls(sp, limit, skip):
    """Wikimedia Commons files from the species' category (Category:<binomial>),
    which Commons curates by scientific name. CC/PD, clean ids, no key. Lower
    volume than iNat/GBIF — a clean top-up for the rare tail."""
    sci = sp.get("sci")
    if not sci: return []
    binomial = " ".join(str(sci).split()[:2])
    base = {"action":"query","format":"json","generator":"categorymembers",
            "gcmtitle":f"Category:{binomial}","gcmtype":"file","gcmlimit":"500",
            "prop":"imageinfo","iiprop":"url|mime","iiurlwidth":"1024"}
    out, cont, pages = [], {}, 0
    while len(out) < limit and pages < 20:
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php",
                             params={**base, **cont}, headers=HEADERS, timeout=TIMEOUT).json()
        except Exception:
            break
        for pg in ((r.get("query") or {}).get("pages") or {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"): continue
            u = ii.get("thumburl") or ii.get("url")
            if u and u not in skip:
                out.append((u, None, None))     # Commons rarely exposes clean coords
        if "continue" in r:
            cont = r["continue"]; pages += 1
        else:
            break
    return out[:limit]

def source_urls(source, sp, limit, skip):
    """Dispatch to a source. Always returns a list; never raises (a flaky source
    must skip, not abort the run)."""
    try:
        if source == "inat":
            return inat_urls(sp["sci"], limit, skip) if sp.get("sci") else []
        if source == "gbif":
            return gbif_urls(sp["taxonKey"], limit, skip)
        if source == "wikimedia":
            return wikimedia_urls(sp, limit, skip)
    except Exception as e:
        print(f"  [{source}] source error ({e}); skipping", flush=True)
    return []

# ---------------- PHASE 1: download ----------------
def dhash(im):
    """64-bit difference hash. Two images with the same dhash are the same photo
    (robust to re-compression/resize), so it catches the 'same picture under two
    different species labels' corruption that exact-byte dedup misses."""
    g = im.convert("L").resize((9, 8))
    px = list(g.getdata())                       # 8 rows x 9 cols
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row*9+col] > px[row*9+col+1] else 0)
    return f"{bits:016x}"

def download_one(url, lat, lon, species, coarse, cls, idx, source):
    try:
        b = requests.get(url, headers=HEADERS, timeout=TIMEOUT); b.raise_for_status()
        im = Image.open(io.BytesIO(b.content)).convert("RGB")
        im.thumbnail((IMG_MAX, IMG_MAX))
    except Exception:
        return None
    stem = f"{idx:08d}"
    try:
        h = dhash(im)
        im.save(INCOMING / f"{stem}.jpg", quality=88)
    except Exception:
        return None
    return dict(stem=stem, species=species, coarse=coarse, cls=cls,
                lat=lat, lon=lon, source_url=url, source=source, img_hash=h)

def source_order(enabled, name, nobox_counts):
    """Default priority, but push any source that has racked up too many no-box
    failures for THIS species to the back, so cleaner sources get tried first."""
    bad = [s for s in enabled if nobox_counts.get((name, s), 0) > NOBOX_REPRIORITIZE]
    if not bad:
        return enabled
    return [s for s in enabled if s not in bad] + bad

def phase_download():
    species = json.loads(Path(SPECIES_JSON).read_text())
    counts = current_counts()
    species.sort(key=lambda s: counts.get(s["name"], 0))

    incoming = json.loads(INCOMING_JSON.read_text()) if INCOMING_JSON.exists() else []
    # Dedup across the WHOLE new dataset, not just the incoming buffer, so a
    # re-run never re-downloads an image we already sorted in a previous pass.
    seen = {r["source_url"] for r in incoming if r.get("source_url")}
    nobox_counts = Counter()      # (species, source) -> no-box failures, for reprioritising
    for fn in ("manifest.json", "no_box_manifest.json"):
        p = OUT / fn
        if p.exists():
            recs = json.loads(p.read_text())
            seen |= {r["source_url"] for r in recs if r.get("source_url")}
            if fn == "no_box_manifest.json":
                for r in recs:
                    nobox_counts[(r.get("species"), r.get("source"))] += 1
    # conflict entries store a LIST of urls (same photo, several species)
    cp = OUT / "conflict_manifest.json"
    if cp.exists():
        for c in json.loads(cp.read_text()):
            seen |= {u for u in c.get("source_urls", []) if u}
    idx = next_idx()
    enabled = list(SOURCES)
    # download enough to NET TARGET_GOOD after boxing: ~TARGET_GOOD / BOX_YIELD.
    print(f"PHASE 1 download -> {OUT}. {len(species)} species, aim {TARGET_GOOD} "
          f"good (~{math.ceil(TARGET_GOOD/BOX_YIELD)} dl). sources: {' -> '.join(enabled)}")

    for si, sp in enumerate(species):
        name, coarse = sp["name"], sp["coarse"]
        try:
            have = counts.get(name, 0)                       # good boxed images so far
            need_good = TARGET_GOOD - have
            if need_good <= 0: continue
            need_dl = math.ceil(need_good / BOX_YIELD)       # downloads to net need_good
            cls = DETECTOR_CLASSES.index(coarse)
            order = source_order(enabled, name, nobox_counts)

            got = 0
            per_source = {}
            # cascade: each source tops up toward need_dl SUCCESSFUL downloads.
            for source in order:
                if got >= need_dl: break
                remaining = need_dl - got
                urls = source_urls(source, sp, remaining * OVERFETCH, seen)
                if not urls: continue
                with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
                    futs = []
                    for (u, la, lo) in urls:
                        futs.append(ex.submit(download_one, u, la, lo, name, coarse, cls, idx, source))
                        idx += 1
                    for f in as_completed(futs):
                        r = f.result()
                        if r:
                            incoming.append(r); seen.add(r["source_url"])
                            got += 1; per_source[source] = per_source.get(source, 0) + 1
                            if got >= need_dl:
                                for fu in futs: fu.cancel()
                                break
                INCOMING_JSON.write_text(json.dumps(incoming, ensure_ascii=False))
            breakdown = " ".join(f"{s}={per_source.get(s,0)}" for s in order)
            print(f"[{si+1}/{len(species)}] {name:<28} have {have:>4}, +{got:>4} dl  ({breakdown})", flush=True)
        except Exception as e:
            # one species failing (network, parsing, whatever) must never abort
            # the overnight run — log it and move on.
            print(f"[{si+1}/{len(species)}] {name:<28} SKIPPED ({e})", flush=True)
            continue
    print(f"\nPHASE 1 done. {len(incoming)} images in {INCOMING}/ awaiting boxing.")

# ---------------- PHASE 2: box ----------------
def phase_box():
    from ultralytics import YOLO
    incoming = json.loads(INCOMING_JSON.read_text()) if INCOMING_JSON.exists() else []
    if not incoming:
        print("nothing in _incoming to box."); return
    print(f"PHASE 2 box -> {OUT}. {len(incoming)} images, model {MODEL}, batch {BATCH}.")
    det = YOLO(MODEL); NAMES = det.names

    manifest = json.loads((OUT/"manifest.json").read_text()) if (OUT/"manifest.json").exists() else []
    nobox = json.loads((OUT/"no_box_manifest.json").read_text()) if (OUT/"no_box_manifest.json").exists() else []
    stats = Counter()
    done_stems = set()

    def predict_batch(paths):
        # a single corrupt/truncated jpg must not kill the whole batch; fall
        # back to per-image prediction so one bad file is skipped, not 32.
        try:
            return det.predict(paths, conf=CONF, imgsz=640, verbose=False)
        except Exception as e:
            print(f"  batch predict failed ({e}); retrying per-image", flush=True)
            out = []
            for p in paths:
                try:
                    out.append(det.predict(p, conf=CONF, imgsz=640, verbose=False)[0])
                except Exception:
                    out.append(None)
            return out

    def flush(batch_recs):
        paths = [str(INCOMING / f"{r['stem']}.jpg") for r in batch_recs]
        results = predict_batch(paths)
        for r, res in zip(batch_recs, results):
            done_stems.add(r["stem"])
            src = INCOMING / f"{r['stem']}.jpg"
            if res is None:                       # prediction failed for this image
                src.unlink(missing_ok=True); stats["discard"]+=1; continue
            W, H = res.orig_shape[1], res.orig_shape[0]
            want = r["coarse"]
            boxes = []
            for b in res.boxes:
                coarse = COCO_TO_COARSE.get(NAMES[int(b.cls)])
                if coarse != want: continue
                x1,y1,x2,y2 = [float(v) for v in b.xyxy[0]]
                cx,cy = ((x1+x2)/2)/W, ((y1+y2)/2)/H
                w,h = (x2-x1)/W, (y2-y1)/H
                if w<=0 or h<=0: continue
                if w>=MAX_BOX_FRAC and h>=MAX_BOX_FRAC: continue   # near-full-frame -> junk
                boxes.append((DETECTOR_CLASSES.index(coarse),cx,cy,w,h))
            if not boxes:
                # YOLO found no acceptable animal -> don't trust that blindly.
                # Could be a real bird it missed; send to manual labeling.
                # (r carries `source`, so phase_download can demote a source that
                #  keeps landing here for a species.)
                src.rename(OUT/"no_box"/f"{r['stem']}.jpg")
                nobox.append(r); stats["manual"]+=1
            else:
                # one OR MORE boxes -> auto-box ALL of them as the queried species.
                # Multiple animals in one GBIF/iNat photo are (almost always) the
                # SAME species, so every box gets the same coarse class. The case
                # that DOES corrupt labels — the same photo tagged as two different
                # species — is caught later by find_conflicts() via img_hash, not
                # here (a coarse detector can't see species).
                split = "val" if int(r["stem"]) % VAL_EVERY == 0 else "train"
                src.rename(OUT/"images"/split/f"{r['stem']}.jpg")
                (OUT/"labels"/split/f"{r['stem']}.txt").write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for (c,cx,cy,w,h) in boxes))
                rec = {k:r.get(k) for k in ("stem","species","coarse","cls","lat","lon","source_url","source","img_hash")}
                rec["split"] = split; rec["n_boxes"] = len(boxes)
                manifest.append(rec)
                stats["single" if len(boxes)==1 else "multi"]+=1

    batch=[]
    for i, r in enumerate(incoming):
        if not (INCOMING/f"{r['stem']}.jpg").exists():
            continue
        batch.append(r)
        if len(batch) >= BATCH:
            flush(batch); batch=[]
            print(f"  boxed {i+1}/{len(incoming)}  singles={stats['single']} "
                  f"multi={stats['multi']} discard={stats['discard']}", flush=True)
    if batch: flush(batch)

    (OUT/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    (OUT/"no_box_manifest.json").write_text(json.dumps(nobox, ensure_ascii=False))
    # keep only records we did NOT box this pass (e.g. file missing) so re-runs
    # are idempotent and don't re-process what's already sorted.
    remaining = [r for r in incoming if r["stem"] not in done_stems]
    INCOMING_JSON.write_text(json.dumps(remaining, ensure_ascii=False))
    print(f"\nPHASE 2 done. +{stats['single']} single-box, {stats['multi']} multi-box, "
          f"{stats['manual']} to manual labeling, {stats['discard']} unreadable. "
          f"manifest now {len(manifest)}.")

# ---------------- conflict detection ----------------
def _hash_of(rec, path):
    """img_hash from the record, else computed from the file on disk (legacy
    records downloaded before we stored hashes)."""
    h = rec.get("img_hash")
    if h: return h
    try:
        with Image.open(path) as im:
            return dhash(im.convert("RGB"))
    except Exception:
        return None

def find_conflicts():
    """Find the SAME photo (identical dhash) assigned to DIFFERENT species — the
    corruption that exact-URL dedup can't catch (one image at two urls, tagged as
    two species). Move every copy to conflicts/ and record it as PRIORITY manual
    labeling. Removes them from train/val so they can't poison training."""
    CONF_JSON = OUT / "conflict_manifest.json"
    manifest = json.loads((OUT/"manifest.json").read_text()) if (OUT/"manifest.json").exists() else []
    nobox = json.loads((OUT/"no_box_manifest.json").read_text()) if (OUT/"no_box_manifest.json").exists() else []
    incoming = json.loads(INCOMING_JSON.read_text()) if INCOMING_JSON.exists() else []
    conflicts = json.loads(CONF_JSON.read_text()) if CONF_JSON.exists() else []
    known = {c["img_hash"] for c in conflicts}

    # (record, kind, file_path) for everything currently placed
    entries = []
    for r in manifest:
        entries.append((r, "manifest", OUT/"images"/r.get("split","train")/f"{r['stem']}.jpg"))
    for r in nobox:
        entries.append((r, "nobox", OUT/"no_box"/f"{r['stem']}.jpg"))
    for r in incoming:
        entries.append((r, "incoming", INCOMING/f"{r['stem']}.jpg"))

    by_hash = defaultdict(list)
    for (r, kind, path) in entries:
        if not path.exists(): continue
        h = _hash_of(r, path)
        if not h: continue
        r["img_hash"] = h                       # cache so we don't re-decode next round
        by_hash[h].append((r, kind, path))

    drop = set()
    new = 0
    for h, group in by_hash.items():
        species = {g[0].get("species") for g in group}
        if len(species) < 2:                    # same (or single) species -> fine
            continue
        # dhash is 64-bit and collides on featureless images (blank sky/water):
        # a low-entropy hash or an implausibly large bucket is a false positive,
        # not one photo. Skip it. (Surviving groups are still PIXEL-verified by
        # dev/verify_conflicts.py before any get labeled.)
        if len(set(h)) <= 3 or len(group) > 8:
            continue
        for (r, kind, path) in group:           # same photo, 2+ species -> CONFLICT
            try: path.rename(OUT/"conflicts"/f"{r['stem']}.jpg")
            except Exception: pass
            if kind == "manifest":              # drop the (now untrustworthy) label
                (OUT/"labels"/r.get("split","train")/f"{r['stem']}.txt").unlink(missing_ok=True)
            drop.add(r["stem"])
        if h not in known:
            conflicts.append({"img_hash": h, "priority": "high",
                              "species": sorted(s for s in species if s),
                              "stems": [g[0]["stem"] for g in group],
                              "source_urls": [g[0].get("source_url") for g in group]})
            new += 1

    if drop:
        (OUT/"manifest.json").write_text(json.dumps([r for r in manifest if r["stem"] not in drop], ensure_ascii=False))
        (OUT/"no_box_manifest.json").write_text(json.dumps([r for r in nobox if r["stem"] not in drop], ensure_ascii=False))
        INCOMING_JSON.write_text(json.dumps([r for r in incoming if r["stem"] not in drop], ensure_ascii=False))
        CONF_JSON.write_text(json.dumps(conflicts, ensure_ascii=False))
    print(f"  conflicts: +{new} new (same photo, 2+ species) -> conflicts/ for PRIORITY labeling; "
          f"{len(conflicts)} total", flush=True)
    return new

# ---------------- loop orchestrator ----------------
def species_under_target():
    """How many species are still below TARGET_GOOD boxed-good images."""
    counts = current_counts()
    species = json.loads(Path(SPECIES_JSON).read_text())
    return sum(1 for s in species if counts.get(s["name"], 0) < TARGET_GOOD)

def run_loop():
    """Download -> box -> recheck counts AFTER boxing; if species are still under
    TARGET_GOOD, download more. Repeat until everything hits target or a round
    makes no progress (sources exhausted for the remaining species)."""
    # If a previous run left downloaded-but-unboxed candidates, BOX THEM FIRST so
    # the deficit (and thus how much to download) is computed from real counts —
    # otherwise we'd re-download ~1000/species on top of what's already waiting.
    pending = json.loads(INCOMING_JSON.read_text()) if INCOMING_JSON.exists() else []
    if pending:
        print(f"\n===== PRE-BOX: {len(pending)} pending candidates from a prior run =====", flush=True)
        phase_box(); find_conflicts()
    for rnd in range(1, MAX_ROUNDS + 1):
        before = len(json.loads((OUT/"manifest.json").read_text())) if (OUT/"manifest.json").exists() else 0
        print(f"\n===== ROUND {rnd}/{MAX_ROUNDS} =====", flush=True)
        phase_download()
        phase_box()
        find_conflicts()          # pull cross-species duplicate photos BEFORE the
                                  # count check, so their species get topped up again
        after = len(json.loads((OUT/"manifest.json").read_text())) if (OUT/"manifest.json").exists() else 0
        under = species_under_target()
        print(f"\n=== round {rnd}: +{after-before} good (total {after}); "
              f"{under} species still under {TARGET_GOOD} ===", flush=True)
        if under == 0:
            print("all species at target — done."); break
        if after == before:
            print("no new good images this round — sources exhausted for the rest. "
                  "remaining gaps need manual labeling (see no_box/)."); break

# ---------------- main ----------------
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("all", "loop"):    run_loop()
    elif phase == "download":       phase_download()
    elif phase == "box":            phase_box()
    elif phase == "conflicts":      find_conflicts()
    else:                           print(f"unknown phase '{phase}' (use: all|download|box|conflicts)")
