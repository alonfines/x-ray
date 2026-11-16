"""Post-run cross-check (not a pre-registered metric): Bonferroni coverage and sizes recomputed
from the raw CSVs with numpy only, independent of conformal_common / bonferrony."""
import json, math
import numpy as np, pandas as pd
D = "/Users/alonfines/Desktop/x-ray/results/outputs/mimic/"
S = "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809"
ALL = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation",
       "Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
CL = {"cardio_pair": ["Enlarged Cardiomediastinum","Cardiomegaly"],
      "lung_core": ["Lung Opacity","Edema","Consolidation"],
      "pleural_free": ["Pneumothorax","Pleural Effusion","Support Devices"],
      "lung_wide": ["Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis"]}
def clean(df):
    y = df[[f"{l}_true_label" for l in ALL]].to_numpy(float); p = df[[f"{l}_prob" for l in ALL]].to_numpy(float)
    k = np.all(np.isin(y,[0.,1.]),1) & np.all(np.isfinite(p),1)
    return df[k]
cal = clean(pd.read_csv(D+S+"_conformal.csv")); te = clean(pd.read_csv(D+S+"_test.csv"))
out = {}
for name, labs in CL.items():
    m = len(labs); a = 0.10/m
    pc = cal[[f"{l}_prob" for l in labs]].to_numpy(float); yc = cal[[f"{l}_true_label" for l in labs]].to_numpy(int)
    sc = np.where(yc==1, 1-pc, pc); n = len(sc); k = math.ceil((n+1)*(1-a))
    thr = np.sort(sc,0)[k-1]
    pt = te[[f"{l}_prob" for l in labs]].to_numpy(float); yt = te[[f"{l}_true_label" for l in labs]].to_numpy(int)
    in1 = (1-pt) <= thr; in0 = pt <= thr          # class-1 / class-0 membership per label
    keep = np.where(yt==1, in1, in0)
    cov = keep.all(1).mean()
    size = (in1.astype(int)+in0.astype(int)).prod(1)
    out[name] = {"n_cal": n, "k": k, "n_test": len(te), "coverage": round(float(cov),6),
                 "mean_size": round(float(size.mean()),4), "median_size": float(np.median(size)),
                 "empty_rate": round(float((size==0).mean()),6)}
print(json.dumps(out, indent=1))
