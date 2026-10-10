"""Resource cost of an ONNX classifier as the plugin runs it: single-query CPU latency, disk, RSS.

Each model is measured in a fresh subprocess so RSS deltas are not polluted by earlier loads.
Usage: perf.py <name>=<classifier_dir> [...]   -> merges into results_perf.json
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
CHILD = r'''
import json, sys, time, resource
import numpy as np
from pathlib import Path
d = Path(sys.argv[1]); texts = json.loads(sys.stdin.read())
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
import onnxruntime as ort
from tokenizers import Tokenizer
t0 = time.perf_counter()
tok = Tokenizer.from_file(str(d / "tokenizer.json")); tok.enable_truncation(128)
f = d / "onnx" / ("model.onnx" if (d / "onnx" / "model.onnx").exists() else "model_quantized.onnx")
sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
load_ms = (time.perf_counter() - t0) * 1000
names = {i.name for i in sess.get_inputs()}
def run(t):
    e = tok.encode(t)
    feed = {"input_ids": np.array([e.ids], dtype=np.int64), "attention_mask": np.array([e.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids], dtype=np.int64)}
    sess.run(None, {k: v for k, v in feed.items() if k in names})
for t in texts[:10]: run(t)
lat = []
for t in texts:
    s = time.perf_counter(); run(t); lat.append((time.perf_counter() - s) * 1000)
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                  "load_ms": load_ms, "rss_mb": (rss1 - rss0) / 1024 / 1024}))
'''


def prompts():
    sys.path.insert(0, str(ROOT))
    from bench import datasets
    d = datasets()
    g, s = d["general"][0], d["swe"][0]
    return g[:150] + s[:50]  # mostly short prompts, some long issue texts (truncated to 128 tokens)


def main():
    out_path = ROOT / "results_perf.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    texts = json.dumps(prompts())
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        path = Path(os.path.expanduser(path))
        r = subprocess.run([sys.executable, "-c", CHILD, str(path)], input=texts, capture_output=True, text=True)
        if r.returncode:
            print(name, "FAILED", r.stderr[-400:]); continue
        m = json.loads(r.stdout.strip().splitlines()[-1])
        f = path / "onnx" / ("model.onnx" if (path / "onnx" / "model.onnx").exists() else "model_quantized.onnx")
        m["disk_mb"] = f.stat().st_size / 1e6
        results[name] = {k: round(v, 1) for k, v in m.items()}
        print(name, results[name])
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
