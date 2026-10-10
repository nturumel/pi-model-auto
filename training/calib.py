"""Cost-aware threshold for the Haiku boundary, chosen on validation and scored on the test sets.

Policy: send to LOW (Haiku) only if P(LOW) >= t; otherwise MEDIUM vs HIGH by argmax (with the 0.45 floor).
Usage: calib.py <classifier_dir> [...]
"""
import sys
import numpy as np
from bench import datasets, onnx_probs

def route(probs, t):
    p = np.where(probs[:, 2] >= probs[:, 1], 2, 1)
    p = np.where(probs.max(-1) < 0.45, 2, p)
    return np.where(probs[:, 0] >= t, 0, p)

def stats(y, p):
    hard = y >= 1
    return {"haiku_share": (p == 0).mean(), "low_recall": (p[~hard] == 0).mean() if (~hard).any() else np.nan,
            "under_to_haiku": (p[hard] == 0).mean(), "severe": (p + 1 < y).mean()}

data = datasets()
for d in sys.argv[1:]:
    pv, pg, ps = (onnx_probs(d, data[s][0]) for s in ["val", "general", "swe"])
    yv, yg, ys = data["val"][1], data["general"][1], data["swe"][1]
    print(f"\n{d}")
    print("   t  | val: haiku% lowRec under sev | test: haiku% lowRec under sev | swe→haiku%")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        v, g, s = stats(yv, route(pv, t)), stats(yg, route(pg, t)), stats(ys, route(ps, t))
        print(f"  {t:.1f} | {100*v['haiku_share']:5.1f} {v['low_recall']:.3f} {v['under_to_haiku']:.3f} {v['severe']:.3f} |"
              f" {100*g['haiku_share']:5.1f} {g['low_recall']:.3f} {g['under_to_haiku']:.3f} {g['severe']:.3f} | {100*s['haiku_share']:5.1f}")
