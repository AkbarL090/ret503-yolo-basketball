"""Bangun dataset_v1 (format YOLO) dari sumber_frame dengan split BLOK frame berurutan (anti-leakage).
Split diambil dari dataset_raw/metadata.csv (sama dengan tugas klasifikasi Pertemuan 3).

  python scripts/build_yolo_dataset.py   ->  dataset_v1/{images,labels}/{train,val,test} + data.yaml
"""
import csv, os, shutil

R, S, O = "dataset_raw", "sumber_frame", "dataset_v1"
split = {}
for r in csv.DictReader(open(f"{R}/metadata.csv")):
    split[r["sumber_frame"]] = {"valid": "val"}.get(r["split"], r["split"])
n = {}
for f, s in split.items():
    stem = os.path.splitext(f)[0]
    for k, src in (("images", f"{S}/images/{f}"), ("labels", f"{S}/labels/{stem}.txt")):
        d = f"{O}/{k}/{s}"; os.makedirs(d, exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, d)
    n[s] = n.get(s, 0) + 1
with open(f"{O}/data.yaml", "w") as fh:
    fh.write(f"path: {os.path.abspath(O)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
             "names:\n  0: backboard\n  1: ball\n  2: rim\n")
print(n)
