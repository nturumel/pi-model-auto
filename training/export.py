"""Export a trained run to the plugin's classifier layout: ONNX (opset 17) + dynamic int8 quantization."""
import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--per-channel", action="store_true")
ap.add_argument("--exclude-classifier", action="store_true", help="keep the classification head in fp32")
ap.add_argument("--fp32", action="store_true", help="ship onnx/model.onnx unquantized instead of int8")
ap.add_argument("--out", default="classifier")
args = ap.parse_args()

best = args.run / "best"
full = args.run / "onnx-full"
main_export(model_name_or_path=str(best), output=full, task="text-classification", opset=17)
out = args.run / args.out
(out / "onnx").mkdir(parents=True, exist_ok=True)
for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    if (best / name).exists():
        (out / name).write_bytes((best / name).read_bytes())
if args.fp32:
    import shutil
    shutil.copy(full / "model.onnx", out / "onnx" / "model.onnx")
    print("exported fp32", out)
    raise SystemExit
extra = {}
if args.exclude_classifier:
    import onnx
    graph = onnx.load(str(full / "model.onnx")).graph
    extra["nodes_to_exclude"] = [n.name for n in graph.node if "classifier" in n.name or "head" in n.name]
quantize_dynamic(str(full / "model.onnx"), str(out / "onnx" / "model_quantized.onnx"),
                 weight_type=QuantType.QInt8, per_channel=args.per_channel, **extra)
print("exported", out)
