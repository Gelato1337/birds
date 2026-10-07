"""
train_sl.py — train ONLY yolo11s and yolo11l on the 313-class species data,
using all 8 GPUs: 4 GPUs per model, both running concurrently via DDP.

  python train_sl.py

s -> GPUs 0,1,2,3   l -> GPUs 4,5,6,7
Each is its own process with DDP across its 4 GPUs (big effective batch).
Logs per model to runs/birdpoke/{s,l}/train_log.txt.
"""
import subprocess, os, sys, time, json
from pathlib import Path

DATA="dataset/data.yaml"; PROJECT="runs/birdpoke"; EPOCHS=120; IMGSZ=640

# model -> (gpu list, per-process GLOBAL batch). 313 classes + 4 MI250X each.
ALLOC = {
    "s": ([0,1,2,3], 256),
    "l": ([4,5,6,7], 128),
}

WORKER = r'''
import sys, json
from pathlib import Path
from ultralytics import YOLO
size, gpus_csv, batch, epochs, imgsz, data, project = sys.argv[1:8]
batch=int(batch); epochs=int(epochs); imgsz=int(imgsz)
gpus=[int(x) for x in gpus_csv.split(",")]
device = list(range(len(gpus))) if len(gpus)>1 else 0

# auto-resume: if an interrupted run left a last.pt, continue from it.
last = Path("runs/detect")/project/size/"weights"/"last.pt"  # ultralytics prepends detect/
if last.exists():
    print(f"resuming {size} from {last}")
    model = YOLO(str(last))
    resume = True
else:
    model = YOLO(f"yolo11{size}.pt")
    resume = False

model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch,
            project=project, name=size, exist_ok=True, resume=resume,
            save_period=5,                 # checkpoint every 5 epochs
            patience=25, optimizer="auto", cos_lr=True, amp=True, cache=False,
            mosaic=1.0, close_mosaic=15, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            fliplr=0.5, degrees=5.0, translate=0.1, scale=0.5,
            device=device, workers=8, seed=42)
m = model.val(data=data, imgsz=imgsz)
res={"size":size,"map50":float(m.box.map50),"map":float(m.box.map)}
if size in ("s","m","n"):
    model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, nms=False)
open(f"{project}/{size}/result.json","w").write(json.dumps(res))
print("DONE", size, res)
'''

def main():
    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    wp = Path(PROJECT)/"_worker_sl.py"; wp.write_text(WORKER)
    procs=[]
    for size,(gpus,batch) in ALLOC.items():
        outdir=Path(PROJECT)/size; outdir.mkdir(parents=True, exist_ok=True)
        logf=open(outdir/"train_log.txt","w")
        env=dict(os.environ)
        gcsv=",".join(map(str,gpus))
        # ROCm: set ONLY ROCR_VISIBLE_DEVICES. Setting CUDA_VISIBLE_DEVICES too
        # makes torch.cuda.is_available() go False on AMD. Clear the others.
        env["ROCR_VISIBLE_DEVICES"]=gcsv
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["MIOPEN_USER_DB_PATH"]=f"/tmp/miopen-{size}"
        Path(env["MIOPEN_USER_DB_PATH"]).mkdir(parents=True, exist_ok=True)
        env["PYTORCH_HIP_ALLOC_CONF"]="expandable_segments:True"
        cmd=[sys.executable, str(wp), size, gcsv, str(batch), str(EPOCHS), str(IMGSZ), DATA, PROJECT]
        p=subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((size,p,logf))
        print(f"launched yolo11{size} on GPUs {gpus} batch {batch} -> {outdir}/train_log.txt")
        time.sleep(4)
    print(f"\nboth training. watch: tail -f {PROJECT}/*/train_log.txt\n")
    for size,p,logf in procs:
        p.wait(); logf.close(); print(f"[{size}] exit {p.returncode}")
    print("\n==== SUMMARY ====")
    for size in ALLOC:
        rf=Path(PROJECT)/size/"result.json"
        if rf.exists():
            r=json.loads(rf.read_text()); print(f"{size}: mAP50={r['map50']:.3f} mAP50-95={r['map']:.3f}")
        else: print(f"{size}: failed — see train_log.txt")

if __name__=="__main__": main()