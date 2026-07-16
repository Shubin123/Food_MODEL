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
2. The image is resized to 224×224 and normalized in JavaScript (canvas).
3. The preprocessed tensor is fed to an ONNX **food classifier** running locally.
4. The predicted dish (one of the 101 Food-101 categories) is mapped to typical
   per-serving calories + macros (see `nutrition.js`) and shown/logged.

---

## The model (what actually ships)

The app bundles a **working, verified** classifier:

| Model | Backbone | Output | Source |
|-------|----------|--------|--------|
| **Swin Food-101** (default) | Swin transformer | 101-class food logits | `onnx-community/swin-finetuned-food101-ONNX` (Apache-2.0) |

Because a public image→calorie *regression* model with usable weights isn't
readily downloadable, the app uses a food **classifier** plus a nutrition
lookup table. The classifier reliably recognizes the dish; `nutrition.js` maps
that dish to approximate per-serving calories and macros.

> `model.onnx` in this folder is the quantized Swin Food-101 checkpoint
> (`onnx/model_quantized.onnx`, ~93 MB). It is downloaded from Hugging Face,
> not committed to git (see `.gitattributes`).

To re-download it:

```bash
curl -L "https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx?download=true" -o model.onnx
```

### Legacy regression option

`convert_to_onnx.py` still exists for exporting a NutritionVerse-Direct /
FoodCNN *regression* checkpoint to ONNX. If you convert one, its spec
(`nutritionverse-direct` / `foodcnn-nutrition5k`) is still available in the
model dropdown and `postprocess` handles the numeric-output path.

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

`model.onnx` (the Swin Food-101 classifier) should sit next to `index.html`.
If it's missing, download it:

```bash
curl -L "https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx?download=true" -o model.onnx
```

Then serve the folder (some browsers block model fetch over `file://`):

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`, paste an image URL (or upload a file), and click
**Analyze**.

### Bring your own regression model (optional)

If you have a NutritionVerse-Direct / FoodCNN checkpoint, convert it to ONNX and
select its architecture in the dropdown:

```bash
pip install torch torchvision
python convert_to_onnx.py --model nutritionverse-direct \
    --checkpoint nv_direct_vit.pth --out model.onnx
```

---

## Input as URL (for testing)

Paste any image URL into the **Photo URL** field to analyze a remote photo
(e.g. a sample from the Nutrition5k/Kaggle datasets). The URL is fetched,
converted to a data URL, and passed through the same preprocessing pipeline as
an uploaded file.

---

## Tests

### Browser unit tests
Open **`test.html`** in a browser and click **Run tests**. They cover:

- `validateImageUrl` (accepts http(s), rejects others/invalid)
- `imageToTensorChannels` (correct `[1,3,H,W]` float layout + normalization)
- `softmax` / `argmax`
- `postprocess` — classifier path (logits → dish → calories/macros) and
  regressor path (numeric outputs, negatives clamped)
- `urlToDataUrl` (mocked `fetch`/`FileReader` network path)

No Node, no npm, no test framework — a ~40-line harness in `test-runner.js`.

### End-to-end model test
`test_e2e.py` runs the **full pipeline headless** against the real `model.onnx`:
fetch an image URL → preprocess → ONNX inference → dish → calories. It reads the
labels/nutrition from `nutrition.js` and the normalization from `models.js`, so
it exercises the shipped constants.

```bash
pip install onnxruntime numpy pillow
python3 test_e2e.py \
  --url "https://upload.wikimedia.org/wikipedia/commons/e/ed/Pink-Frosted-Donut.jpg" \
  --expect donuts
```

Expected output ends with the donut's calories and `PASS`:

```
  Detected: Donuts (100.0%)
  Calories: 452 kcal
  ...
PASS
```

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | App UI; loads ORT from CDN |
| `nutrition.js` | Food-101 labels + per-serving calorie/macro lookup table |
| `models.js` | Pure logic: validation, preprocessing, softmax/argmax, postprocessing, model specs |
| `app.js` | ORT session loading, inference, photo input, food log |
| `model.onnx` | Swin Food-101 classifier (downloaded, git-ignored) |
| `convert_to_onnx.py` | PyTorch checkpoint → ONNX (legacy regression option) |
| `model-specs.json` | Reference model definitions |
| `test_e2e.py` | Headless end-to-end test (URL → calories) against `model.onnx` |
| `test.html` / `test-runner.js` / `tests.js` | Browser test harness |
| `styles.css` | Styling |


## futher readings and ideas to be considered

[Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf) This model may use depth which is available on many iphones through ir.

[NutritionVerse-Direct: Exploring Deep Neural Networks for Multitask Nutrition
Prediction from Food Images](https://arxiv.org/pdf/2405.07814) Vit model



