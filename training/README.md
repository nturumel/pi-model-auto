# Classifier training and evaluation

Everything here runs locally with open models and Apache-2.0 data.

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# Train (same split, labels and cost-sensitive loss as train_cost_sensitive.py; ~40 s on Apple silicon)
./venv/bin/python train.py --base sentence-transformers/all-MiniLM-L6-v2 --lr 8e-5 --epochs 4 \
  --risk-lambda 0.5 --output runs/minilm

# Export to ONNX: per-channel int8 is the shipped default; --fp32 writes onnx/model.onnx instead
./venv/bin/python export.py runs/minilm --per-channel

# Quality on the benchmark test split and SWE-bench Verified, plus routing cost
./venv/bin/python bench.py quality onnx runs/minilm/classifier --name minilm
./venv/bin/python bench.py table

# Single-query CPU latency, disk and RSS; Haiku-boundary threshold sweep chosen on validation
./venv/bin/python perf.py minilm=runs/minilm/classifier
./venv/bin/python calib.py runs/minilm/classifier
```

`bench.py` also evaluates the length heuristic (`heuristic`), frozen embeddings with logistic
regression (`embed-lr <model>`), and zero-shot local LLMs (`llm <model> --subset 300`).

To ship a new model, copy `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` and `onnx/` into `models/complexity-classifier/`, then update the release tag
and SHA-256 values in `scripts/install-classifier.mjs`.
