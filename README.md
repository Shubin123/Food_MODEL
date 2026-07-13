# Food Tracker

A **free, fully client-side** web app that estimates **calories and macros**
(calories, mass, protein, fat, carbohydrates) from a photo of food, using a
**real nutrition model** that runs **100% on the user's device** — no server,
no API keys, no Node.

Inference is done with **[ONNX Runtime Web](https://onnxruntime.ai/)** loaded
from a CDN, so the model executes in the browser via WebGL/WASM.

---

## How it works

1. You give it a photo — by **URL** (handy for testing) or by **file upload**.
2. The image is resized/cropped and normalized in JavaScript (canvas).
3. The preprocessed tensor is fed to an ONNX nutrition model running locally.
4. The 5 regression outputs are mapped to calories + macros and shown/logged.

---

## Model choice (research summary)

| Model | Backbone | Heads (regression) | Year | Note |
|-------|----------|--------------------|------|------|
| **NutritionVerse-Direct** | ViT-base-patch16-224 | calories, mass, protein, fat, carbs | 2024 | **SOTA**; combined MAE 412.6 on NV-Real, ~25% better than Inception-ResNet |
| **FoodCNN / Nutrition5k** | Inception-ResNet | calories, mass, fat, carbs, protein | 2021 (CVPR) | MIT-licensed, easiest to retrain |

- **Nutrition5k** (Thames et al., CVPR 2021, Google) — `https://github.com/google-research-datasets/Nutrition5k`. First large, accurately-annotated generic-food dataset; the direct-prediction architecture (Inception-ResNet + FC heads) is the de-facto baseline for this task.
- **NutritionVerse-Direct** (Keller et al., 2024, arXiv:2405.07814, U. Waterloo / NRC) — shows a **ViT** backbone beats Inception-ResNet for direct nutrition regression, predicting all five attributes from a single RGB image (trained on NV-Real with **no channel normalization**). Checkpoints: `bit.ly/genai4good`.
- **NutritionVerse / NV-Real / NV-Synth** datasets: `https://www.kaggle.com/nutritionverse/datasets`.

### Why ONNX Runtime Web (not TF.js)
Both ORT Web and TF.js run in-browser. ORT Web was chosen because the SOTA
models are published as **PyTorch** checkpoints, and `torch.onnx.export`
produces an ONNX graph that ORT Web runs with broad op coverage and WebGL/WebGPU
acceleration. TF.js would require an extra PyTorch→TF→TFJS conversion chain.

---

## Run it (no build step)

The model is **not** checked in (it's a large binary). You have two options:

### Option A — bring your own `.onnx` (recommended)
1. Get a checkpoint:
   - NutritionVerse-Direct: request the released weights at `bit.ly/genai4good`.
   - FoodCNN: clone `https://github.com/FoodCNN/FoodCNN`.
2. Convert it to ONNX:
   ```bash
   pip install torch torchvision
   python convert_to_onnx.py --model nutritionverse-direct \
       --checkpoint nv_direct_vit.pth --out model.onnx
   ```
   (Adjust the fully-connected widths in `convert_to_onnx.py` if your
   checkpoint's layer sizes differ — the script will tell you on a mismatch.)
3. Put `model.onnx` next to `index.html`.
4. Serve the folder (some browsers block model fetch over `file://`):
   ```bash
   python3 -m http.server 8000
   ```
   then open `http://localhost:8000`.

### Option B — try without a real model
`index.html` + `app.js` + `models.js` run standalone; only `Analyze` requires
`model.onnx`. Open `test.html` to run the unit tests.

---

## Input as URL (for testing)

Paste any image URL into the **Photo URL** field to analyze a remote photo
(e.g. a sample from the Nutrition5k/Kaggle datasets). The URL is fetched,
converted to a data URL, and passed through the same preprocessing pipeline as
an uploaded file.

---

## Tests

Open **`test.html`** in a browser and click **Run tests**. They cover:

- `validateImageUrl` (accepts http(s), rejects others/invalid)
- `imageToTensorChannels` (correct `[1,3,H,W]` float layout + normalization)
- `postprocess` (raw outputs → calories/macros, negatives clamped)
- `urlToDataUrl` (mocked `fetch`/`FileReader` network path)

No Node, no npm, no test framework — a ~40-line harness in `test-runner.js`.

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | App UI; loads ORT from CDN |
| `models.js` | Pure logic: validation, preprocessing, postprocessing, model specs |
| `app.js` | ORT session loading, inference, photo input, food log |
| `convert_to_onnx.py` | PyTorch checkpoint → ONNX (NutritionVerse-Direct / FoodCNN) |
| `model-specs.json` | Reference model definitions |
| `test.html` / `test-runner.js` / `tests.js` | Browser test harness |
| `styles.css` | Styling |
