# Food Tracker: accuracy and cleanup implementation plan

## Objective

Replace the current **food-classifier + fixed typical serving** estimate with a
measured, editable nutrition-estimation workflow. The product should be honest
about uncertainty, improve portion estimates, and remain a small static web app
unless a server is proven necessary.

## Progress

- **Completed first implementation/test loop (2026-07-18):** added an editable
  gram portion control that scales the detected food's reference nutrition,
  session invalidation when the selected model changes, and an automated
  Puppeteer gate. The gate verifies the upload-to-log flow, calculation,
  session reload, responsiveness of the deterministic fixture, and browser unit
  tests. It does **not** establish real-model accuracy; that starts at M1 with
  weighed held-out photos.

## Why the current result is poor

The model that actually ships is `swin-food101`, a 101-class image classifier.
After selecting one label, `nutrition.js` returns exactly one hard-coded serving
and macro profile for that label. Therefore the calorie result does not depend
on the food quantity in the photo. A correctly recognised pizza slice and an
entire pizza can receive the same estimate. The result also cannot account for
multiple foods, ingredients, recipes, cooking oil, or branded products.

Other correctness and maintainability issues to resolve:

- The interface and README describe a nutrition-regression model, but the
  default/only usable artifact is the classifier.
- The model selector does not reload an already cached session when its model
  or model URL changes; the displayed selection can differ from the session
  that produced the result.
- Images are stretched to a square instead of being resized/cropped according
  to model preprocessing; no EXIF orientation handling or input-quality check
  exists.
- The nutrition table has no source, version, density, serving definition, or
  provenance. Fixed values mix per-serving weights and calories.
- A single top-1 class hides ambiguity and makes correction impossible.
- `convert_to_onnx.py` is not a reproducible exporter for released models:
  architecture/checkpoint compatibility is guessed and the claimed
  `torchvision.models.inception_resnet_v2` is not a standard torchvision model.
- Browser tests largely test helpers and mocks, not the model contract,
  regression accuracy, calibration, accessibility, or the user correction
  workflow. The Python E2E test depends on a remote image and a locally
  downloaded binary, so it is not a deterministic CI gate.

## Research findings and product implications

1. **Portion is the decisive missing signal.** Nutrition5k contains total and
   ingredient-level weights/nutrients, multi-view RGB video, overhead RGB-D,
   and official evaluation code; its authors present depth as an improvement to
   nutrition prediction. A single RGB image intrinsically has weak scale and
   volume information. Make quantity a first-class estimate and request either
   a reference/second view or a user correction. [Nutrition5k paper](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html), [dataset and evaluation details](https://github.com/google-research-datasets/Nutrition5k).
2. **A direct multitask regressor is a better baseline than a serving table,
   not a guarantee of accurate real-world tracking.** NutritionVerse-Direct's
   ViT regression architecture predicts calories, mass, protein, fat, and
   carbohydrates jointly, reporting a 412.6 combined MAE on NV-Real, a 25.5%
   improvement over its Inception-ResNet comparison. It should be evaluated on
   our target photos before being adopted. [NutritionVerse-Direct](https://arxiv.org/abs/2405.07814).
3. **Train for the actual capture conditions.** NutritionVerse provides
   84,984 synthetic images with depth and masks plus 889 real images/251 dishes;
   its study explicitly supports synthetic pretraining followed by real-image
   fine-tuning. Nutrition5k is also geographically/menu limited, so neither
   benchmark alone validates the app's users or cuisines. [NutritionVerse
   study](https://arxiv.org/abs/2309.07704), [NutritionVerse-Real](https://arxiv.org/abs/2401.08598).
4. **Use a nutrition source at the ingredient level.** USDA FoodData Central
   provides searchable and downloadable food/nutrient data. Do not place a
   public API key in this static client: use a curated/versioned local subset
   first, or a server-side proxy only if live branded-food lookup is approved.
   [FoodData Central API guide](https://fdc.nal.usda.gov/api-guide/),
   [downloadable datasets](https://fdc.nal.usda.gov/download-datasets/).

## Success metrics and non-negotiable guardrails

Define these before changing models. Record both the existing baseline and each
candidate on an immutable, held-out test set of representative user photos with
weighed ground truth.

| Area | Metric | Initial release target |
| --- | --- | --- |
| Calories | MAE, median absolute percentage error, signed bias | At least 30% lower calorie MAE than current fixed-serving baseline on the target set; publish the measured value, not an unverified claim. |
| Portion | mass MAE / MAPE | Report separately by single item, mixed plate, and reference-assisted captures. |
| Recognition | top-1/top-3 accuracy and confusion matrix | Measure by cuisine/category; low confidence must lead to correction, not a confident calorie. |
| Calibration | interval coverage and expected calibration error | An 80% prediction interval should contain roughly 80% of held-out values. |
| Product | correction rate and post-correction error | Log consented, de-identified corrections; do not silently train on them. |
| Safety | user messaging | Label every output as an estimate and state it is unsuitable for medical dosing or diagnosis. |

Set the final numeric acceptance thresholds after a baseline evaluation. Reject
a candidate that improves an aggregate metric while materially regressing on a
high-use cohort or cuisine.

## Implementation sequence

### 0. Freeze a reproducible baseline (first)

1. Add `docs/` with an architecture diagram, model/data provenance, licenses,
   and a decision record explaining the present classifier-plus-serving fallback.
2. Build a private or consented evaluation manifest: image ID, capture mode,
   dish/components, weighed grams, authoritative nutrient source, calories and
   macros. Split by meal/user, never by near-duplicate photo.
3. Add a local benchmark command that emits CSV/JSON metrics, per-cohort slices,
   signed error, and charts. Check in only manifests/fixtures that are licensed
   and small; keep food photos and model weights outside git or in approved
   object storage.
4. Run the current Swin/table pipeline against that set and commit its results
   as the baseline. Verify Food-101 label ordering against the exact ONNX
   artifact and record the model SHA-256.

**Exit criterion:** the team can reproduce a baseline report and identify the
largest errors by recognition, portion, food density, and meal complexity.

### 1. Deliver an honest, useful hybrid UX without waiting for a new model

1. Replace the single hard-coded lookup with a versioned food catalog containing
   per-100 g nutrients, source ID/date, preparation state, density range, and
   aliases. Generate the small client JSON from a documented source dataset;
   do not hand-maintain 101 unrelated serving assumptions.
2. Show top-3 recognition candidates and confidence; let the user choose or
   search a food. For a mixed plate, support multiple components rather than
   forcing one Food-101 label.
3. Ask for a portion in grams, household measure, or serving count. Default to
   a clearly labelled estimate/range and recalculate calories/macros from
   `grams * nutrients-per-100g / 100`. Make every detected field editable.
4. Add an optional capture assist: plate size/reference object selection and a
   guided second angle. Do not claim metric volume from a single unreferenced
   photo. If available and consented, offer device depth input as an optional
   enhancement.
5. Render an estimate range and confidence explanation (e.g., low recognition,
   unknown cooking method, no scale reference). Persist the source, selected
   food, user edits, model/version, and range with each log entry.
6. Add accessible controls, keyboard paths, error states, image-size/type
   limits, EXIF orientation correction, aspect-ratio-preserving centre crop,
   and explicit CORS guidance for URL input.

**Exit criterion:** users can correct food and quantity in under 20 seconds;
the logged calculation is traceable and reproducible from its saved inputs.

### 2. Build the model pipeline and select a candidate by evidence

1. Create a separate Python training/evaluation environment (`requirements.txt`
   or `pyproject.toml` with pinned versions) rather than adding a frontend
   framework. Include data download instructions, checksums, licensing,
   deterministic seeds, and dataset adapters for NutritionVerse and Nutrition5k.
2. Implement a simple, reproducible baseline first: image encoder + five
   non-negative regression heads for mass, kcal, protein, fat, and carbs.
   Train with log/robust targets, target normalization, augmentations matching
   mobile photos, and a weighted multitask loss. Validate calorie consistency
   against the Atwater relationship as a diagnostic, without overwriting labels.
3. Fine-tune a ViT direct regressor initialized from the published
   NutritionVerse approach, then adapt it on a small, representative,
   properly-consented local capture set. Compare it with the current table
   baseline and a component-based estimate; do not rely solely on paper scores.
4. Add optional branches only after the RGB baseline: segmentation/component
   detection; two-view fusion; depth/reference-scale volume estimation; and
   density-informed ingredient calculations. Ablate each branch to verify that
   it earns its added capture burden and complexity.
5. Produce calibrated uncertainty using an ensemble, quantile heads, or
   conformal calibration on a held-out calibration split. Evaluate coverage by
   food type and capture mode.
6. Export only the selected, tested model to ONNX. Add an ONNX parity test
   comparing PyTorch and ONNX outputs for fixed fixtures, document exact tensor
   names/shapes/preprocessing/output order, quantize only after checking error,
   and publish a model card with limitations and evaluation slices.

**Exit criterion:** one candidate beats the fixed-serving baseline on the
pre-agreed target metrics, its uncertainty is calibrated, and its browser ONNX
output is parity-tested.

### 3. Integrate the selected model safely in the web app

1. Replace the free-form architecture/model controls with a versioned model
   manifest. A model change must invalidate/reload the session; cache by model
   URL, checksum, execution provider, and preprocessing version.
2. Split the browser code into small ES modules: `model-session`, `image-input`,
   `preprocess`, `nutrition-catalog`, `estimator`, `storage`, `ui`, and
   `analytics` (if consented). Keep framework-free HTML/CSS/JS unless module
   bundling genuinely becomes necessary.
3. Implement a model-output schema that includes component estimates, intervals,
   confidence/quality flags, model/catalog versions, and error-safe validation.
   Never turn malformed or negative output into a plausible zero silently.
4. Use Web Worker inference when it keeps interaction responsive; prefer
   WebGPU where supported with a WASM fallback and measure both. Version the
   ONNX Runtime CDN URL and provide offline/cache behavior.
5. Keep nutrition lookup client-side for the curated catalog. If live USDA or
   branded search is later approved, add a minimal server/proxy that keeps API
   credentials off-device, rate-limits requests, caches permitted responses,
   and documents privacy/data retention.

**Exit criterion:** switching models is correct, inference remains responsive,
and every displayed result is editable, versioned, and accompanied by its range.

### 4. Clean up the project and make quality repeatable

1. Remove or quarantine obsolete paths only after replacement coverage exists:
   the hand-authored serving table, misleading regression selector, and
   unverified converter. Preserve a tagged baseline or `legacy/` note until the
   new benchmark proves the replacement.
2. Correct UTF-8 mojibake in UI/README and reconcile all copy with the actual
   shipped model. State on the landing page whether the result is classifier
   fallback, direct regression, or user-corrected calculation.
3. Add a minimal `package.json` only for linting, formatting, and browser/unit
   tests (for example, ESLint + Prettier + a lightweight runner). Do not add
   React, a state library, or a bundler merely to reorganise this static app.
   Pin and lock dev dependencies; retain a no-build production path if modules
   are served directly.
4. Replace remote-only tests with local image/tensor fixtures. Add tests for
   aspect-ratio/orientation preprocessing, model manifest/session invalidation,
   output schema validation, per-100-g arithmetic, edit persistence, interval
   rendering, catalog provenance, and keyboard/accessibility checks.
5. Add CI for formatter/linter, unit tests, model contract/parity tests, and
   benchmark-report generation on a small licensed fixture set. Keep expensive
   training and full benchmark runs manual or scheduled with recorded artifacts.
6. Add `CONTRIBUTING.md`, `docs/model-card.md`, `docs/data-governance.md`, and
   a concise README with setup, model download/cache, testing, privacy, known
   limitations, and rollback instructions.

**Exit criterion:** a clean clone can run the app, test deterministic fixtures,
and explain exactly where every estimate, nutrient value, and model binary came
from.

## Proposed milestones

1. **M1 - Baseline report:** evaluation manifest, current measured error, model
   provenance, and prioritized error taxonomy.
2. **M2 - Correctable tracker:** sourced per-100-g catalog, top-k food choice,
   quantity editor, multi-item log, and range/disclaimer; this is the earliest
   user-facing improvement.
3. **M3 - Candidate model:** reproducible training/evaluation pipeline and
   evidence-based selection of direct RGB regression versus the hybrid fallback.
4. **M4 - Assisted portion:** optional second view/reference/depth branch only
   if its ablation improves target metrics enough to justify user friction.
5. **M5 - Production cleanup:** ONNX manifest/parity, modules, deterministic
   tests/CI, model card, data governance, and removal of retired code.

## Decisions to make before M2/M3

- Define the intended user/capture context (home meals, packaged foods,
  restaurants, cuisines, camera platforms) and whether a second photo or
  reference object is acceptable.
- Approve how ground-truth photos and corrections are collected, stored,
  retained, and consented; health/nutrition data requires a privacy review.
- Choose whether the product stays strictly offline with a curated catalog, or
  whether a backend is authorised for food search and protected credentials.
- Agree on the error threshold at which the app must prompt for manual entry
  rather than offering an automated estimate.
