"""Convert the FP32 BirdNET ONNX to FP16 (keep float32 I/O so the app feeds the
same [1,144000] float32 in / gets float32 out — no JS change). Then verify the
top detections on our sample clips still match before we ship it."""
import onnx, json, numpy as np, librosa, onnxruntime as ort
from onnxconverter_common import float16

m = onnx.load("model.onnx")
m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
onnx.save(m16, "birdnet_fp16.onnx")
import os
print("fp32 %.1f MB -> fp16 %.1f MB" % (os.path.getsize("model.onnx")/1e6, os.path.getsize("birdnet_fp16.onnx")/1e6))

labels = [l.strip() for l in open("BirdNET_GLOBAL_6K_V2.4_Labels.txt", encoding="utf-8") if l.strip()]
amap = {int(k): v for k, v in json.load(open("../../birdnet_map.json")).items()}
sess = ort.InferenceSession("birdnet_fp16.onnx", providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
W = 144000
for f in ["Dendrocopos_major.m4a", "Erithacus_rubecula.wav", "Cyanistes_caeruleus.wav"]:
    audio, _ = librosa.load("../audio_samples/" + f, sr=48000, mono=True)
    best = {}
    for s in range(0, max(1, len(audio)), W):
        c = audio[s:s+W]
        if len(c) < W: c = np.pad(c, (0, W-len(c)))
        logits = np.array(sess.run(None, {name: c[None, :].astype(np.float32)})[0])[0]
        conf = 1/(1+np.exp(-logits.astype(np.float32)))
        for idx, sp in amap.items():
            if conf[idx] > best.get(sp, 0): best[sp] = float(conf[idx])
    top = sorted(best.items(), key=lambda x: -x[1])[:2]
    print(f"  {f}: " + ", ".join(f"{sp} {c:.3f}" for sp, c in top))
