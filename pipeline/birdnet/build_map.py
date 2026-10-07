"""Build birdnet_map.json: BirdNET class index -> our species name, matched by
scientific binomial. Lets the app run BirdNET (6522 global classes) and keep only
the detections that correspond to our 313 Finnish species. Ships to repo root."""
import json
from pathlib import Path

sp = json.loads(Path("../../species.json").read_text())
labels = [l.strip() for l in open("BirdNET_GLOBAL_6K_V2.4_Labels.txt", encoding="utf-8") if l.strip()]
print("labels:", len(labels))

def binom(s): return " ".join(str(s).split()[:2]).lower()

bn = {}                                   # birdnet binomial -> class index
for i, l in enumerate(labels):
    bn[binom(l.split("_")[0])] = i

mapping, unmatched = {}, []
for s in sp:
    idx = bn.get(binom(s.get("sci", "")))
    if idx is not None:
        mapping[idx] = s["name"]
    else:
        unmatched.append((s["name"], s.get("coarse")))

Path("../../birdnet_map.json").write_text(json.dumps(mapping, ensure_ascii=False))
birds_un = [n for n, c in unmatched if c == "bird"]
print(f"mapped {len(mapping)} of {len(sp)} species")
print(f"unmatched: {len(unmatched)}  (mammals {sum(1 for _,c in unmatched if c=='mammal')}, birds {len(birds_un)})")
print("unmatched BIRDS (BirdNET doesn't know / name mismatch):", birds_un[:25])
