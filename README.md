# Food Tracker

A small, client-side food photo tracker. Choose an image, review the detected
food and its estimated nutrition, adjust the portion in grams, then add it to
the local log.

## What the estimate means

The shipped Swin Food-101 model identifies one of 101 food categories in the
browser. The app then uses a local reference nutrition profile for that food.
It does **not** measure calories or portion size directly from a photograph.
Always review the detected food and grams before logging; this is not medical
or dietary advice.

No photo, food log, or model inference is sent to an application server. The
model is downloaded from Hugging Face and cached by the browser for later use.

## Run locally

Serve the repository over HTTP, then open the local address in a browser:

```bash
python -m http.server 8000
```

## Test

```bash
npm ci
npm test
```

The browser test uses a deterministic ONNX Runtime stub. It verifies upload,
inference flow, portion editing, logging, cache/session invalidation, and the
browser unit suite without downloading the production model.

## Model

The default model is
[`onnx-community/swin-finetuned-food101-ONNX`](https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX)
(Apache-2.0). The application loads its quantized ONNX file directly from
Hugging Face.
