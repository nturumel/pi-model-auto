import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { finished } from "node:stream/promises";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const modelRoot = join(packageRoot, "models", "complexity-classifier");
// classifier-v2: all-MiniLM-L6-v2 fine-tuned with training/train.py, per-channel int8 (23 MB).
const releaseBase = "https://github.com/nturumel/pi-model-auto/releases/download/classifier-v2";
const assets = [
  { name: "config.json", path: "config.json", sha256: "1b047cd031360252e9926e895bc439edea6455eebd71542302f3e39cf1e97108" },
  { name: "tokenizer.json", path: "tokenizer.json", sha256: "79ea220416a57ca8acdf069d71875d5d2ddadef66021aeb6c120a7c546c04344" },
  { name: "tokenizer_config.json", path: "tokenizer_config.json", sha256: "20fc02202d48143b228b3acbd9507eb33096bef2b705f2106148eaa6d4c6f134" },
  { name: "special_tokens_map.json", path: "special_tokens_map.json", sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a" },
  { name: "model_quantized.onnx", path: "onnx/model_quantized.onnx", sha256: "c09ee80d433403b597f0c06fe2d491682e9690ab6c0c7cae0fce517fa51ea582" },
];

async function digest(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}
let complete = true;
for (const asset of assets) {
  try {
    if (await digest(join(modelRoot, asset.path)) !== asset.sha256) complete = false;
  } catch {
    complete = false;
  }
}
if (complete) process.exit(0);

const stagingRoot = `${modelRoot}.tmp-${process.pid}`;
await rm(stagingRoot, { recursive: true, force: true });
try {
  for (const asset of assets) {
    const destination = join(stagingRoot, asset.path);
    await mkdir(dirname(destination), { recursive: true });
    const response = await fetch(`${releaseBase}/${asset.name}`, { redirect: "follow" });
    if (!response.ok || !response.body) throw new Error(`download failed for ${asset.name}: HTTP ${response.status}`);
    const output = createWriteStream(destination, { flags: "wx" });
    const hash = createHash("sha256");
    for await (const chunk of response.body) {
      hash.update(chunk);
      if (!output.write(chunk)) await new Promise((resolve) => output.once("drain", resolve));
    }
    output.end();
    await finished(output);
    const actual = hash.digest("hex");
    if (actual !== asset.sha256) throw new Error(`checksum mismatch for ${asset.name}: ${actual}`);
  }
  await rm(modelRoot, { recursive: true, force: true });
  await rename(stagingRoot, modelRoot);
  console.log("Installed pi-model-auto local complexity classifier");
} catch (error) {
  await rm(stagingRoot, { recursive: true, force: true });
  throw error;
}
