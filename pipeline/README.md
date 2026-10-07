# Pipeline

The scripts that build the dataset, label it and train **fieldmark-s**. Run them in
order from a working directory that holds `species.json` and the `dataset/` folder.
The Supabase scripts need `SUPABASE_URL` and a server-side `SUPABASE_KEY` (secret key)
in the environment; never commit that key.

## 1. Data — `data/`

| Script | What it does |
|---|---|
| `build_species.py` | Builds `species.json`: Finnish birds and mammals with enough photographed GBIF observations |
| `add_missing_species.py` | Adds rarer species below the observation cutoff (resolved via GBIF) |
| `fetch_v2.py` · `run_fetch.sh` | Downloads photos (iNaturalist → GBIF → Wikimedia), auto-boxes them with YOLO11x, tops up thin species, flags the same photo filed under two species |
| `fetch_wiki.py` | Fetches the Wikipedia species profiles shown in the app (`wiki_species.json`) |

## 2. Labeling — `labeling/`

| Script | What it does |
|---|---|
| `verify_conflicts.py` | Confirms cross-species duplicates pixel by pixel (drops hash collisions) |
| `populate_labeler.py` | Uploads the confirmed conflicts to the labeling queue used by `labeler.html` |
| `pull_conflicts.py` | Applies the human decisions back into the dataset (re-boxes and files each photo under the chosen species) |

## 3. Training — `training/` (Slurm, written for LUMI's AMD GPUs)

| Script | What it does |
|---|---|
| `remap.py` · `remap.sh` | Rewrites labels from placeholder classes to species indices and writes `data.yaml` |
| `train_sl.py` · `train_sl.sh` | Trains YOLO11s and YOLO11l in parallel (4 GPUs each), validates, exports the small model to ONNX |

Set `#SBATCH --account` before submitting. The exported ONNX is copied to the repo root as
`fieldmark-s-vN.onnx`.

## 4. Sound — `birdnet/`

| Script | What it does |
|---|---|
| `build_map.py` | Maps BirdNET's 6,522 global classes onto our species by scientific name (`birdnet_map.json`) |
| `to_fp16.py` | Converts BirdNET to FP16 for the browser and checks its predictions are unchanged |
