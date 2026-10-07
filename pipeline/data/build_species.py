# =====================================================================
# build_species_list.py — enumerate ALL Finnish birds + mammals from GBIF
# =====================================================================
# Produces species.json: every bird/mammal species with research-grade,
# photographed, georeferenced observations in Finland above a threshold.
# This replaces the hand-typed SPECIES table (and fixes the wrong-taxonKey
# 0-results problem — keys come straight from GBIF here).
#
#   pip install requests tqdm
#   python build_species_list.py

import json
import requests
from tqdm import tqdm

GBIF = "https://api.gbif.org/v1"

# Direct GBIF REST call for species names — bypasses pygbif's name_usage,
# which mangles kwargs into requests on some versions (the 'name' bug).
def gbif_name(species_key):
    try:
        d = requests.get(f"{GBIF}/species/{species_key}", timeout=15).json()
        vern = d.get("vernacularName")
        sci = d.get("scientificName", "") or d.get("canonicalName", "")
        if not vern:
            vr = requests.get(f"{GBIF}/species/{species_key}/vernacularNames",
                              timeout=15).json().get("results", [])
            vern = next((v["vernacularName"] for v in vr
                         if v.get("language") == "eng"), None)
        return (vern or sci or f"sp_{species_key}"), sci
    except Exception:
        return f"sp_{species_key}", ""

COUNTRY = "FI"
MIN_OBS = 40                 # drop vagrants/one-offs not worth a class
# Stable GBIF backbone class taxonKeys (no lookup needed — avoids the
# name_backbone kwarg bug). class label -> (GBIF classKey, our coarse label)
CLASSES = {
    212: ("Aves", "bird"),       # birds
    359: ("Mammalia", "mammal"), # mammals
}

def facet_species(class_key):
    """Count obs per speciesKey via GBIF occurrence facets (direct REST)."""
    counts = {}
    offset = 0
    while True:
        params = {
            "country": COUNTRY, "taxonKey": class_key,
            "mediaType": "StillImage", "hasCoordinate": "true",
            "facet": "speciesKey", "facetLimit": 1000,
            "facetOffset": offset, "limit": 0,
        }
        r = requests.get(f"{GBIF}/occurrence/search", params=params,
                         timeout=30).json()
        facets = r.get("facets", [])
        if not facets: break
        buckets = facets[0].get("counts", [])
        if not buckets: break
        for b in buckets:
            counts[int(b["name"])] = b["count"]
        if len(buckets) < 1000: break
        offset += 1000
    return counts

species = []
for class_key, (class_name, coarse) in CLASSES.items():
    print(f"\nenumerating {class_name} ({coarse})…")
    counts = facet_species(class_key)
    keep = {k: c for k, c in counts.items() if c >= MIN_OBS}
    print(f"  {len(counts)} species total, {len(keep)} with >= {MIN_OBS} obs")
    for skey, c in tqdm(sorted(keep.items(), key=lambda x: -x[1]), desc="names"):
        name, sci = gbif_name(skey)
        species.append(dict(name=name, sci=sci, taxonKey=skey,
                            coarse=coarse, obs=c))

species.sort(key=lambda s: (s["coarse"], -s["obs"]))
with open("species.json", "w") as f:
    json.dump(species, f, indent=2, ensure_ascii=False)

birds = sum(1 for s in species if s["coarse"] == "bird")
mams  = sum(1 for s in species if s["coarse"] == "mammal")
print(f"\n✅ species.json written: {birds} birds + {mams} mammals = {len(species)} classes")
print("Top by observations:")
for s in species[:12]:
    print(f"  {s['obs']:>7}  {s['coarse']:<7}{s['name']}")