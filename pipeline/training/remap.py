"""
remap.py — rewrite every YOLO label's class from the coarse placeholder (0/1) to the
real species index (0..N-1), using manifest (stem -> species) + species.json (order).
Box coordinates are untouched; also writes the N-class data.yaml.

Parallelized for the big dataset — run on a COMPUTE node (sbatch remap.sh).
266k small-file rewrites are too heavy/metadata-bound for the login node.
"""
import json, os
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(".")
DS = ROOT / "dataset"
species = json.loads((ROOT / "species.json").read_text())
manifest = json.loads((DS / "manifest.json").read_text())
NAME_TO_IDX = {s["name"]: i for i, s in enumerate(species)}   # species.json order == class id
STEM_TO_SP  = {m["stem"]: m["species"] for m in manifest}

def remap_one(p):
    sp = STEM_TO_SP.get(Path(p).stem)
    if sp is None or sp not in NAME_TO_IDX:
        return 0
    cls = str(NAME_TO_IDX[sp])
    out = []
    for line in Path(p).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(); parts[0] = cls      # swap class, keep box coords
        out.append(" ".join(parts))
    if not out:
        return 0
    Path(p).write_text("\n".join(out) + "\n")
    return 1

if __name__ == "__main__":
    files = []
    for split in ("train", "val"):
        d = DS / "labels" / split
        if d.is_dir():
            files += [str(p) for p in d.glob("*.txt")]
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"remapping {len(files)} label files with {n} workers…", flush=True)
    with Pool(n) as pool:
        done = sum(pool.map(remap_one, files, chunksize=256))
    names = [s["name"] for s in species]
    (DS / "data.yaml").write_text(
        f"path: {DS.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {json.dumps(names, ensure_ascii=False)}\n")
    print(f"remapped {done}/{len(files)} labels to species classes; "
          f"wrote data.yaml (nc={len(names)})", flush=True)
