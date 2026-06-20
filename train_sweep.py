"""
BirdPoke — YOLO detector training sweep (n / s / m / l / x).

Trains any subset of YOLO11 sizes so you can benchmark the speed/accuracy
tradeoff on YOUR data and devices, then exports the light ones to ONNX for
the browser.

Usage:
    python train_sweep.py --sizes n s m            # quick, edge-friendly set
    python train_sweep.py --sizes n s m l x        # full sweep (needs a big GPU)
    python train_sweep.py --sizes s --epochs 80    # one model, fewer epochs
    python train_sweep.py --sizes x --resume       # resume an interrupted run

Hardware notes are in the README. Short version:
    RTX 3070 (10GB): n/s/m fine. l is tight (small batch). x won't fit at 640
                     unless you drop batch to ~2 and use AMP — painfully slow.
    Colab T4 (16GB): n/s/m/l ok; x slow but possible.
    Colab A100 / LUMI MI250X: everything, fast.

Env:
    pip install ultralytics==8.3.* onnx onnxslim onnxruntime
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"
PROJECT   = "runs/birdpoke"
IMGSZ     = 640  # default; override with --imgsz

# Per-size batch defaults sized for a 10GB GPU (RTX 3070). YOLO streams images
# from disk — it does NOT load the whole dataset into memory, so dataset size
# does not cause OOM; BATCH SIZE does. If you still OOM, lower batch further
# or drop --imgsz to 512. On a big GPU (A100/MI250X) you can raise these a lot.
SIZE_PRESETS = {
    "n": dict(batch=16, export_onnx=True),
    "s": dict(batch=12, export_onnx=True),    # primary browser model
    "m": dict(batch=8,  export_onnx=True),    # browser ceiling / strong server
    "l": dict(batch=4,  export_onnx=False),   # server
    "x": dict(batch=2,  export_onnx=False),   # server reference, max accuracy
}


def train_one(size, epochs, batch, resume, export_onnx, imgsz):
    name = f"yolo11{size}"
    print(f"\n========== {name}  (batch={batch}, epochs={epochs}) ==========")
    model = YOLO(f"{name}.pt")     # COCO-pretrained start = far faster convergence

    model.train(
        data=DATA_YAML, epochs=epochs, imgsz=imgsz, batch=batch,
        project=PROJECT, name=size, exist_ok=True, resume=resume,
        patience=25, optimizer="auto", cos_lr=True,
        amp=True,                  # mixed precision — big VRAM + speed win
        cache=False,               # set "ram"/"disk" if you have spare memory
        mosaic=1.0, close_mosaic=15,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, degrees=5.0, translate=0.1, scale=0.5,
        device=0, workers=8, seed=42,
    )

    m = model.val(data=DATA_YAML, imgsz=imgsz)
    row = dict(size=size, map50=float(m.box.map50), map=float(m.box.map))
    print(f"[{size}] mAP50={row['map50']:.3f}  mAP50-95={row['map']:.3f}")

    if export_onnx:
        p = model.export(format="onnx", imgsz=imgsz, opset=17,
                         simplify=True, dynamic=False, nms=False)
        print(f"[{size}] ONNX -> {p}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["s"],
                    choices=list(SIZE_PRESETS))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="lower to 512 or 448 if you OOM (quadratic memory saving)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the per-size batch preset")
    args = ap.parse_args()

    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    results = []
    for size in args.sizes:
        cfg = SIZE_PRESETS[size]
        batch = args.batch if args.batch else cfg["batch"]
        results.append(train_one(size, args.epochs, batch,
                                 args.resume, cfg["export_onnx"], args.imgsz))

    print("\n================ SWEEP SUMMARY ================")
    print(f"{'size':<6}{'mAP50':>9}{'mAP50-95':>11}")
    for r in results:
        print(f"{r['size']:<6}{r['map50']:>9.3f}{r['map']:>11.3f}")
    print("Pick the smallest size whose mAP is 'good enough' for your photos.")


if __name__ == "__main__":
    main()