# =====================================================================
# add_missing_species.py — append the LuontoPortti species your
# species.json was missing (below the MIN_OBS=40 cutoff), via GBIF.
# =====================================================================
# Run this in the same folder as species.json. It:
#   1. reads your existing species.json
#   2. for each missing binomial, resolves the GBIF species key + the
#      Finnish photographed-observation count (so 'obs' matches your data)
#   3. appends any not already present (idempotent — safe to re-run)
#   4. rewrites species.json, re-sorted like build_species_list.py
#
#   pip install requests tqdm
#   python add_missing_species.py
#
# The list below was produced by diffing your species.json against the
# LuontoPortti bird + mammal lists, with genus-rename false positives
# already removed (e.g. Great Egret was already present as Ardea alba).

import json, re
from pathlib import Path
import requests
from tqdm import tqdm

GBIF = "https://api.gbif.org/v1"
COUNTRY = "FI"

# (scientific binomial, coarse class). 61 birds + 16 mammals = 77 species.
MISSING = [
    # ---- birds ----
    ("Polysticta stelleri", "bird"), ("Circus macrourus", "bird"),
    ("Recurvirostra avosetta", "bird"), ("Uria aalge", "bird"),
    ("Milvus migrans", "bird"), ("Upupa epops", "bird"),
    ("Gallinago media", "bird"), ("Phylloscopus proregulus", "bird"),
    ("Phylloscopus trochiloides", "bird"), ("Larus hyperboreus", "bird"),
    ("Calidris canutus", "bird"), ("Lymnocryptes minimus", "bird"),
    ("Calidris falcinellus", "bird"), ("Accipiter gentilis", "bird"),
    ("Lullula arborea", "bird"), ("Serinus serinus", "bird"),
    ("Loxia leucoptera", "bird"), ("Oriolus oriolus", "bird"),
    ("Alcedo atthis", "bird"), ("Calidris ferruginea", "bird"),
    ("Somateria spectabilis", "bird"), ("Tringa stagnatilis", "bird"),
    ("Aythya marila", "bird"), ("Anthus cervinus", "bird"),
    ("Porzana porzana", "bird"), ("Anthus petrosus", "bird"),
    ("Merops apiaster", "bird"), ("Calidris maritima", "bird"),
    ("Aix sponsa", "bird"), ("Cygnus atratus", "bird"),
    ("Chlidonias niger", "bird"), ("Falco peregrinus", "bird"),
    ("Circus pygargus", "bird"), ("Cygnus columbianus", "bird"),
    ("Rissa tridactyla", "bird"), ("Alle alle", "bird"),
    ("Schoeniclus pusillus", "bird"), ("Charadrius dubius", "bird"),
    ("Tachybaptus ruficollis", "bird"), ("Schoeniclus rusticus", "bird"),
    ("Falco vespertinus", "bird"), ("Branta ruficollis", "bird"),
    ("Xenus cinereus", "bird"), ("Crex crex", "bird"),
    ("Alca torda", "bird"), ("Anser indicus", "bird"),
    ("Pluvialis squatarola", "bird"), ("Acanthis hornemanni", "bird"),
    ("Falco rusticolus", "bird"), ("Eremophila alpestris", "bird"),
    ("Bubo scandiacus", "bird"), ("Streptopelia turtur", "bird"),
    ("Coturnix coturnix", "bird"), ("Poecile palustris", "bird"),
    ("Linaria flavirostris", "bird"),
    # ---- mammals ----
    ("Castor fiber", "mammal"), ("Myodes rufocanus", "mammal"),
    ("Mustela putorius", "mammal"), ("Sorex caecutiens", "mammal"),
    ("Microtus arvalis", "mammal"), ("Sicista betulina", "mammal"),
    ("Mus musculus", "mammal"), ("Microtus oeconomus", "mammal"),
    ("Myopus schisticolor", "mammal"), ("Neovison vison", "mammal"),
    ("Sorex isodon", "mammal"), ("Apodemus agrarius", "mammal"),
    ("Myodes rutilus", "mammal"), ("Lemmus lemmus", "mammal"),
    ("Neomys fodiens", "mammal"), ("Sus scrofa", "mammal"),
]


def match_species_key(binomial):
    """Resolve a binomial to a GBIF backbone speciesKey."""
    r = requests.get(f"{GBIF}/species/match",
                     params={"name": binomial, "strict": "false"},
                     timeout=15).json()
    # prefer an accepted SPECIES-rank usage key
    key = r.get("usageKey")
    return key, r.get("scientificName", binomial), r.get("canonicalName", binomial)


def english_name(species_key, fallback):
    try:
        vr = requests.get(f"{GBIF}/species/{species_key}/vernacularNames",
                          timeout=15).json().get("results", [])
        en = next((v["vernacularName"] for v in vr
                   if v.get("language") == "eng"), None)
        return en or fallback
    except Exception:
        return fallback


def fi_obs_count(species_key):
    """Finnish photographed, georeferenced observation count (matches your data)."""
    try:
        r = requests.get(f"{GBIF}/occurrence/search", timeout=20, params={
            "country": COUNTRY, "taxonKey": species_key,
            "mediaType": "StillImage", "hasCoordinate": "true", "limit": 0,
        }).json()
        return r.get("count", 0)
    except Exception:
        return 0


def binom(sci):
    toks = re.sub(r"[(),]", " ", sci).split(); out = []
    for t in toks:
        if out and (t[0].isupper() or t[0].isdigit()): break
        out.append(t)
        if len(out) == 2: break
    return " ".join(out)


def main():
    path = Path("species.json")
    species = json.loads(path.read_text())
    have_keys = {s["taxonKey"] for s in species}
    have_binoms = {binom(s["sci"]) for s in species}

    added = 0
    for name_bin, coarse in tqdm(MISSING, desc="resolving"):
        if name_bin in have_binoms:
            continue  # already present under this binomial
        key, sci, canon = match_species_key(name_bin)
        if not key or key in have_keys:
            continue
        common = english_name(key, canon or name_bin)
        obs = fi_obs_count(key)
        species.append(dict(name=common, sci=sci or name_bin,
                            taxonKey=key, coarse=coarse, obs=obs))
        have_keys.add(key); have_binoms.add(name_bin)
        added += 1

    species.sort(key=lambda s: (s["coarse"], -(s.get("obs") or 0)))
    path.write_text(json.dumps(species, indent=2, ensure_ascii=False))

    birds = sum(1 for s in species if s["coarse"] == "bird")
    mams = sum(1 for s in species if s["coarse"] == "mammal")
    print(f"\n✅ added {added} species. species.json now: "
          f"{birds} birds + {mams} mammals = {len(species)} total.")


if __name__ == "__main__":
    main()