"""Ukur waktu per gambar (praproses + inferensi + pascaproses, batch 1, dengan warm-up).
  python scripts/benchmark.py E1_pretrained:416 E3_imgsz640:640
"""
import glob, sys
from ultralytics import YOLO

fs = sorted(glob.glob("dataset_v1/images/val/*.jpg"))
for item in sys.argv[1:]:
    name, sz = item.split(":"); sz = int(sz)
    m = YOLO(f"runs/detect/{name}/weights/best.pt")
    for f in fs[:10]:
        m.predict(f, imgsz=sz, verbose=False)                      # warm-up
    t = [sum(m.predict(f, imgsz=sz, verbose=False)[0].speed.values()) for f in fs * 5]
    print(f"{name} imgsz={sz}: total ms/gambar rata-rata {sum(t)/len(t):.1f} | median {sorted(t)[len(t)//2]:.1f}")
