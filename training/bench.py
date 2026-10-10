"""One harness for every routing paradigm: quality on general + coding test sets, and routing cost.

Usage:
  bench.py quality <kind> <path-or-id> [--name N]   kind: heuristic | onnx | hf | embed-lr | llm
  bench.py table                                     summarize results/*.json

Labels: LOW / MEDIUM / HIGH. With the current router config MEDIUM and HIGH both run on Sonnet, so the
binary LOW (Haiku) vs not-LOW split is what drives cost and the main quality risk.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATASET_ID = "anasnassar/llm-query-complexity-benchmark"
DATASET_REVISION = "e111dc811aea9103010db741db9af0505240e77c"
LABELS = ["LOW", "MEDIUM", "HIGH"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWE_DIFFICULTY = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 2}
# Measured per-turn cost from live omp runs (router.log usage.cost.total).
TURN_COST = {"haiku": 0.0008, "sonnet": 0.0225}
CONFIDENCE_FLOOR = 0.45  # same fail-upward rule as the plugin


# ---------- data ----------
def _bench(split):
    p = hf_hub_download(DATASET_ID, f"{split}.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def datasets():
    train = _bench("train")
    tr, va = train_test_split(train, test_size=0.2, random_state=42, stratify=[r["ground_truth"] for r in train])
    general = _bench("test")
    from datasets import load_dataset
    swe = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    swe_rows = [{"text": "Fix this issue in the repository:\n\n" + r["problem_statement"],
                 "y": SWE_DIFFICULTY[r["difficulty"]]} for r in swe]
    as_xy = lambda rows: ([r["text"] for r in rows], np.array([L2I[r["ground_truth"]] for r in rows]))
    return {"train": as_xy(tr), "val": as_xy(va), "general": as_xy(general),
            "swe": ([r["text"] for r in swe_rows], np.array([r["y"] for r in swe_rows]))}


# ---------- metrics ----------
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def route(probs):
    p = probs.argmax(-1)
    return np.where(probs.max(-1) < CONFIDENCE_FLOOR, 2, p)


def metrics(y, probs):
    p = route(probs)
    hard = y >= 1
    low_share = float((p == 0).mean())
    out = {
        "acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=[0, 1, 2]),
        "high_recall": float((p[y == 2] == 2).mean()) if (y == 2).any() else math.nan,
        "severe_under": float((p + 1 < y).mean()),
        # binary Haiku boundary
        "haiku_share": low_share,
        "under_to_haiku": float((p[hard] == 0).mean()) if hard.any() else math.nan,   # real task sent to Haiku
        "missed_savings": float((p[~hard] != 0).mean()) if (~hard).any() else math.nan,  # trivial sent to Sonnet
        "cost_per_1k": 1000 * (low_share * TURN_COST["haiku"] + (1 - low_share) * TURN_COST["sonnet"]),
    }
    if (~hard).any() and hard.any():
        from sklearn.metrics import roc_auc_score
        out["low_auc"] = float(roc_auc_score((~hard).astype(int), probs[:, 0]))
    return {k: round(float(v), 4) for k, v in out.items()}


# ---------- paradigms ----------
def heuristic_probs(texts):
    # Upstream fallback: lastUserLen weight 0.5 normalized over 120..1200 chars (context/tools absent).
    score = np.array([min(1, max(0, (len(t) - 120) / 1080)) * 0.5 for t in texts])
    bucket = np.where(score < 0.3, 0, np.where(score < 0.52, 1, 2))
    return np.eye(3)[bucket]


def onnx_probs(model_dir, texts, bs=32):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    onnx_file = model_dir / "onnx" / ("model.onnx" if (model_dir / "onnx" / "model.onnx").exists() else "model_quantized.onnx")
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="np")
        out.append(sess.run(None, {k: v for k, v in enc.items() if k in names})[0])
    return softmax(np.concatenate(out))


def hf_probs(model_dir, texts, bs=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.float().cpu().numpy())
    return softmax(np.concatenate(out))


def embed(model_id, texts, bs=64, query_prefix=""):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="mps")
    m.max_seq_length = 128
    return m.encode([query_prefix + t for t in texts], batch_size=bs, normalize_embeddings=True, show_progress_bar=False)


def embed_lr_probs(model_id, data, prefix=""):
    from sklearn.linear_model import LogisticRegression
    xtr, ytr = data["train"]; xva, yva = data["val"]
    etr, eva = embed(model_id, xtr, query_prefix=prefix), embed(model_id, xva, query_prefix=prefix)
    best = None
    for c in [0.25, 0.5, 1, 2, 4, 8]:
        clf = LogisticRegression(C=c, max_iter=3000).fit(etr, ytr)
        f1 = f1_score(yva, clf.predict(eva), average="macro")
        if best is None or f1 > best[0]: best = (f1, c, clf)
    clf = best[2]
    return {split: clf.predict_proba(embed(model_id, data[split][0], query_prefix=prefix))
            for split in ["general", "swe"]}, {"C": best[1], "val_f1": round(best[0], 4)}


LLM_PROMPT = (
    "You route requests for a coding assistant to a model tier.\n"
    "LOW = trivial or factual, a small fast model is enough.\n"
    "MEDIUM = ordinary task needing some reasoning or a moderate code change.\n"
    "HIGH = hard: deep reasoning, design, proofs, tricky debugging, or multi-step work.\n\n"
    "Request:\n{q}\n\nAnswer with one word: LOW, MEDIUM or HIGH.")


def llm_probs(model_id, texts, limit_chars=600):
    """Zero-shot: one forward pass, compare next-token logits for the three label words."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(dev).eval()
    label_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["LOW", "MEDIUM", "HIGH"]]
    out = []
    with torch.no_grad():
        for t in texts:
            msgs = [{"role": "user", "content": LLM_PROMPT.format(q=t[:limit_chars])}]
            kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            ids = tok(prompt, return_tensors="pt").to(dev)
            logits = model(**ids).logits[0, -1].float().cpu().numpy()
            out.append(logits[label_ids])
    return softmax(np.array(out))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--subset", type=int, default=0, help="evaluate on a fixed random subset per test set")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if a.cmd == "table":
        return table()

    data = datasets()
    if a.subset:
        rng = np.random.default_rng(0)
        for split in ["general", "swe"]:
            x, y = data[split]
            idx = np.sort(rng.choice(len(y), min(a.subset, len(y)), replace=False))
            data[split] = ([x[i] for i in idx], y[idx])
    extra = {}
    t0 = time.time()
    if a.kind == "embed-lr":
        probs, extra = embed_lr_probs(a.path, data, a.prefix)
    else:
        fn = {"heuristic": lambda p, x: heuristic_probs(x), "onnx": onnx_probs, "hf": hf_probs, "llm": llm_probs}[a.kind]
        probs = {split: fn(a.path, data[split][0]) for split in ["general", "swe"]}
    res = {"name": a.name or Path(str(a.path)).name, "kind": a.kind, "path": a.path, "subset": a.subset,
           "eval_secs": round(time.time() - t0), **extra,
           "general": metrics(data["general"][1], probs["general"]), "swe": metrics(data["swe"][1], probs["swe"])}
    (RESULTS / f"{res['name']}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def table():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    perf = {}
    pf = ROOT / "results_perf.json"
    if pf.exists(): perf = json.loads(pf.read_text())
    hdr = f"{'name':22s} {'kind':9s} | {'gen F1':>6s} {'gen acc':>7s} {'HIGHrec':>7s} {'sev':>5s} | {'swe F1':>6s} {'swe→H%':>6s} {'sweUnd':>6s} {'sweAUC':>6s} | {'$/1k swe':>8s} | {'p50ms':>5s} {'MB':>5s} {'RSS':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["general"]["macro_f1"]):
        g, s = r["general"], r["swe"]; pr = perf.get(r["name"], {})
        print(f"{r['name'][:22]:22s} {r['kind'][:9]:9s} | {g['macro_f1']:6.3f} {g['acc']:7.3f} {g['high_recall']:7.3f} {g['severe_under']:5.3f} | "
              f"{s['macro_f1']:6.3f} {100*s['haiku_share']:6.1f} {s['under_to_haiku']:6.3f} {s.get('low_auc', float('nan')):6.3f} | "
              f"{s['cost_per_1k']:8.2f} | {pr.get('p50_ms', float('nan')):5.1f} {pr.get('disk_mb', float('nan')):5.0f} {pr.get('rss_mb', float('nan')):5.0f}")


if __name__ == "__main__":
    main()
