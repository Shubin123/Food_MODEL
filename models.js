/*
 * Food Tracker - model logic (no frameworks, no build step).
 *
 * Runs a real nutrition model (e.g. NutritionVerse-Direct / FoodCNN, converted
 * to ONNX) entirely in the browser via ONNX Runtime Web. This file holds the
 * pure, testable pieces: input validation, image preprocessing, and output
 * postprocessing. Model definitions live in MODEL_SPECS.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});

  // Each spec describes how to feed an ONNX nutrition model and how to read
  // its outputs. Tweak MEAN/STD and `outputs` order to match the checkpoint
  // you convert (see convert_to_onnx.py and README).
  const MODEL_SPECS = {
    // SOTA 2024 (Keller et al.) ViT-base-patch16-224 + 5 regression heads.
    // Paper trains on NV-Real with NO channel normalization -> mean/std = 0/1.
    // If predictions look off, switch to ImageNet norm:
    //   mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    'nutritionverse-direct': {
      id: 'nutritionverse-direct',
      label: 'NutritionVerse-Direct (ViT, 2024)',
      size: 224,
      mean: [0, 0, 0],
      std: [1, 1, 1],
      inputName: null, // use session.inputNames[0]
      // Order matches the 5 regression heads:
      outputs: [
        { name: null, key: 'calories', unit: 'kcal' },
        { name: null, key: 'mass', unit: 'g' },
        { name: null, key: 'protein', unit: 'g' },
        { name: null, key: 'fat', unit: 'g' },
        { name: null, key: 'carbs', unit: 'g' }
      ]
    },
    // FoodCNN / Nutrition5k (Thames et al., CVPR 2021) Inception-ResNet.
    // Uses standard ImageNet normalization.
    'foodcnn-nutrition5k': {
      id: 'foodcnn-nutrition5k',
      label: 'FoodCNN / Nutrition5k (Inception-ResNet)',
      size: 299,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      inputName: null,
      outputs: [
        { name: null, key: 'calories', unit: 'kcal' },
        { name: null, key: 'mass', unit: 'g' },
        { name: null, key: 'fat', unit: 'g' },
        { name: null, key: 'carbs', unit: 'g' },
        { name: null, key: 'protein', unit: 'g' }
      ]
    }
  };

  // Validates a user-supplied image URL (the "input as URL" testing feature).
  function validateImageUrl(url) {
    if (typeof url !== 'string' || url.trim() === '') {
      throw new Error('Image URL is required');
    }
    let parsed;
    try {
      parsed = new URL(url, global.location ? global.location.href : undefined);
    } catch (err) {
      throw new Error('Invalid URL');
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('URL must use http(s)');
    }
    return parsed.href;
  }

  // Converts an RGBA pixel buffer (from canvas getImageData) into a
  // channel-first Float32Array normalized for the model. Pure + testable.
  function imageToTensorChannels(pixels, size, mean, std) {
    const count = size * size;
    const data = new Float32Array(3 * count);
    const m0 = mean[0], m1 = mean[1], m2 = mean[2];
    const s0 = std[0] || 1, s1 = std[1] || 1, s2 = std[2] || 1;
    let r = 0, g = count, b = 2 * count;
    for (let i = 0; i < pixels.length; i += 4) {
      data[r++] = (pixels[i] / 255 - m0) / s0;
      data[g++] = (pixels[i + 1] / 255 - m1) / s1;
      data[b++] = (pixels[i + 2] / 255 - m2) / s2;
    }
    return { data, dims: [1, 3, size, size] };
  }

  // Draws an <img>/<canvas> into a square canvas of `size` and returns the
  // preprocessed tensor. Uses the browser canvas API.
  function preprocessImage(imageEl, spec) {
    const size = spec.size || 224;
    const canvas = global.document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageEl, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size);
    return imageToTensorChannels(imgData.data, size, spec.mean, spec.std);
  }

  // Maps raw regression outputs (numbers, in spec order) to a nutrition object.
  // Clamps negatives to 0 and rounds to sensible precision.
  function postprocess(raw, spec) {
    const out = {};
    spec.outputs.forEach((o, idx) => {
      let v = Array.isArray(raw) ? raw[idx] : raw[o.name];
      v = typeof v === 'number' && isFinite(v) ? v : 0;
      v = Math.max(0, v);
      out[o.key] = o.key === 'calories' ? Math.round(v) : Math.round(v * 10) / 10;
      out[o.key + 'Unit'] = o.unit;
    });
    return out;
  }

  // Resolves a model spec by id (falls back to the SOTA default).
  function getSpec(id) {
    return MODEL_SPECS[id] || MODEL_SPECS['nutritionverse-direct'];
  }

  FT.models = {
    MODEL_SPECS,
    validateImageUrl,
    imageToTensorChannels,
    preprocessImage,
    postprocess,
    getSpec
  };
})(typeof window !== 'undefined' ? window : globalThis);
