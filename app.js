/*
 * Food Tracker - UI + ONNX Runtime Web inference (no frameworks, no build step).
 * Loads a converted nutrition model (.onnx) and runs it entirely client-side.
 * Depends on: models.js (FT.models) and the global `ort` from the CDN script.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});
  const M = FT.models;
  const STORE_KEY = 'food-tracker-entries';

  let session = null;
  let activeSpec = null;
  let sessionKey = null;

  function $(id) { return document.getElementById(id); }

  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
    catch (e) { return []; }
  }
  function writeStore(entries) { localStorage.setItem(STORE_KEY, JSON.stringify(entries)); }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read image'));
      reader.readAsDataURL(blob);
    });
  }
  function fileToDataUrl(file) { return blobToDataUrl(file); }

  // Fetches a remote image URL and returns a data URL for the model.
  async function urlToDataUrl(url) {
    const valid = M.validateImageUrl(url);
    const res = await fetch(valid);
    if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
    return blobToDataUrl(await res.blob());
  }

  // Loads the ONNX model once and caches the inference session.
  // Uses persistent Cache API when available so the model binary is only
  // downloaded once; subsequent visits load from cache.
  async function loadModel(modelUrl, specId, onProgress) {
    if (!global.ort) throw new Error('ONNX Runtime Web failed to load');
    activeSpec = M.getSpec(specId);
    const nextKey = `${modelUrl}|${activeSpec.id}`;
    if (session && sessionKey === nextKey) return session;

    const CM = FT.modelCache;
    let buffer = null;

    // 1. Try persistent browser cache first.
    if (CM && CM.available) {
      try {
        if (await CM.isCached(modelUrl)) {
          console.log('[app] Cache HIT for', modelUrl);
          buffer = await CM.loadFromCache(modelUrl);
        } else {
          console.log('[app] Cache MISS for', modelUrl);
        }
      } catch (e) { /* fall through to network */ }
    }

    // 2. Cache miss – download and store for next time.
    if (!buffer && CM && CM.available) {
      try {
        buffer = await CM.downloadAndCache(modelUrl, onProgress);
      } catch (e) {
        // Download/cache failed (network error, 404, quota, etc.).
        // Fall through to let ORT fetch the model directly.
      }
    }

    // 3. If we still don't have a buffer (cache unavailable or download failed),
    //    let ORT handle the fetch directly.
    if (!buffer) {
      session = await global.ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm', 'webgl'],
        graphOptimizationLevel: 'all'
      });
      sessionKey = nextKey;
      return session;
    }

    // 4. Create session from the ArrayBuffer (cached or freshly downloaded).
    session = await global.ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm', 'webgl'],
      graphOptimizationLevel: 'all'
    });
    sessionKey = nextKey;
    return session;
  }

  async function analyze() {
    const status = $('status');
    setStatus('Preparing…');
    $('resultCard').classList.add('hidden');

    const url = $('imageUrl').value.trim();
    const file = $('fileInput').files && $('fileInput').files[0];
    let imageEl;
    try {
      if (file) {
        const objUrl = URL.createObjectURL(file);
        imageEl = await loadImage(objUrl);
        showPreview(objUrl);
      } else if (url) {
        const dataUrl = await urlToDataUrl(url);
        imageEl = await loadImage(dataUrl);
        showPreview(url);
      } else {
        return setStatus('Provide a photo URL or upload a file.', true);
      }
    } catch (err) {
      return setStatus(err.message, true);
    }

    try {
      const modelUrl = $('modelUrl').value.trim() || 'https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx';
      const specId = currentSpecId();
      if (!session || sessionKey !== `${modelUrl}|${specId}`) {
        setStatus('Loading model…');
        const progBar = $('progressBar');
        const progFill = $('progressFill');
        const wasCached = FT.modelCache && FT.modelCache.available &&
          await FT.modelCache.isCached(modelUrl);
        if (!wasCached && progBar) progBar.classList.remove('hidden');
        await loadModel(modelUrl, specId, function onProg(_a) {
          var pct = (_a && _a.percent) || 0;
          if (progFill) { progFill.style.width = pct + '%'; progFill.textContent = pct + '%'; }
        });
        if (progBar) progBar.classList.add('hidden');
        updateCacheStatus();
      }
      const pre = M.preprocessImage(imageEl, activeSpec);
      const tensor = new global.ort.Tensor('float32', pre.data, pre.dims);
      const feeds = {};
      feeds[session.inputNames[0]] = tensor;

      const t0 = performance.now();
      const results = await session.run(feeds);
      const dt = Math.round(performance.now() - t0);

      let raw;
      if (activeSpec.type === 'classifier') {
        // Classifier: one output tensor holding all class logits.
        const outName = session.outputNames[0];
        raw = Array.from(results[outName].data);
      } else {
        const names = (activeSpec.outputs || []).map((o) => o.name).filter(Boolean);
        const outNames = names.length ? names : session.outputNames;
        if (outNames.length === 1) {
          raw = Array.from(results[outNames[0]].data);
        } else {
          raw = outNames.map((n) => results[n].data[0]);
        }
      }
      const nutrition = M.postprocess(raw, activeSpec);
      nutrition.model = activeSpec.label;
      nutrition.inferenceMs = dt;
      renderResult(nutrition);
      setStatus(`Ran locally in ${dt} ms · ${activeSpec.label}`);
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  function currentSpecId() {
    const sel = $('modelId');
    return sel && sel.value ? sel.value : 'swin-food101';
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not load image'));
      img.src = src;
    });
  }

  function renderResult(n) {
    $('resultCard').classList.remove('hidden');
    const rows = [];
    if (n.name) {
      const conf = typeof n.confidence === 'number' ? ` (${n.confidence}%)` : '';
      rows.push(['Detected', `${n.name}${conf}`]);
    }
    rows.push(
      ['Calories', `${n.calories} kcal`],
      ['Mass', `${n.mass} g`],
      ['Protein', `${n.protein} g`],
      ['Fat', `${n.fat} g`],
      ['Carbs', `${n.carbs} g`]
    );
    $('resultBody').innerHTML = rows
      .map(([k, v]) => `<p><strong>${k}:</strong> <span>${v}</span></p>`)
      .join('');
    FT._lastResult = n;
    const editor = $('portionEditor');
    const portion = $('portionGrams');
    if (n.mass > 0) {
      portion.value = String(Math.round(n.mass));
      editor.classList.remove('hidden');
      $('estimateNote').textContent = n.label
        ? 'Calories and macros are scaled from the detected food’s reference nutrition. Review the grams before logging.'
        : 'Calories and macros are model estimates. Adjusting grams scales this result proportionally.';
    } else {
      editor.classList.add('hidden');
    }
  }

  function roundNutrition(n) {
    n.calories = Math.round(n.calories);
    ['mass', 'protein', 'fat', 'carbs'].forEach((key) => { n[key] = Math.round(n[key] * 10) / 10; });
    return n;
  }

  function updatePortion() {
    const current = FT._lastResult;
    if (!current) return;
    const grams = Number($('portionGrams').value);
    if (!Number.isFinite(grams) || grams <= 0) {
      setStatus('Portion must be greater than 0 grams.', true);
      return;
    }
    let next;
    if (current.label && FT.nutrition) {
      next = FT.nutrition.nutritionForLabelAndMass(current.label, grams);
    } else {
      const factor = grams / current.mass;
      next = {
        calories: current.calories * factor, mass: grams,
        protein: current.protein * factor, fat: current.fat * factor, carbs: current.carbs * factor
      };
    }
    Object.assign(current, roundNutrition(next));
    renderResult(current);
  }

  function showPreview(src) {
    const img = $('resultImg');
    img.src = src;
    img.classList.remove('hidden');
  }

  function logEntry() {
    if (!FT._lastResult) return;
    const n = FT._lastResult;
    const entries = readStore();
    entries.push({
      calories: n.calories, mass: n.mass, protein: n.protein,
      fat: n.fat, carbs: n.carbs, name: n.name || null,
      model: n.model, at: new Date().toISOString()
    });
    writeStore(entries);
    FT._lastResult = null;
    $('resultCard').classList.add('hidden');
    $('imageUrl').value = '';
    if ($('fileInput')) $('fileInput').value = '';
    renderEntries();
  }

  function renderEntries() {
    const entries = readStore();
    const total = entries.reduce((s, e) => s + (Number(e.calories) || 0), 0);
    $('totalCalories').textContent = `${total} kcal`;
    const ul = $('entries');
    ul.innerHTML = '';
    entries.slice().reverse().forEach((e) => {
      const li = document.createElement('li');
      const title = e.name ? `${e.name} - ${e.calories} kcal` : `${e.calories} kcal`;
      li.innerHTML =
        `<div><strong>${title}</strong>` +
        `<div class="entry-meta">P ${e.protein} · F ${e.fat} · C ${e.carbs} g · ${new Date(e.at).toLocaleString()}</div></div>`;
      ul.appendChild(li);
    });
  }

  function setStatus(msg, isError) {
    const el = $('status');
    el.textContent = msg;
    el.className = 'status' + (isError ? ' error' : '');
  }

  /** Updates the cache-status indicator next to the model URL input. */
  async function updateCacheStatus() {
    var statusEl = $('cacheStatus');
    var dlBtn = $('downloadBtn');
    if (!statusEl) return;

    var CM = FT.modelCache;
    if (!CM || !CM.available) {
      statusEl.textContent = '';
      statusEl.className = 'cache-status';
      if (dlBtn) dlBtn.classList.add('hidden');
      return;
    }

    var modelUrl = ($('modelUrl').value || '').trim() || 'https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx';
    var cached = false;
    try { cached = await CM.isCached(modelUrl); } catch (e) { /* ignore */ }

    if (cached) {
      var sizeBytes = null;
      try { sizeBytes = await CM.getCachedSize(modelUrl); } catch (e) { /* ignore */ }
      var sizeStr = sizeBytes ? (sizeBytes / (1024 * 1024)).toFixed(1) + ' MB' : 'cached';
      statusEl.textContent = '✓ Cached (' + sizeStr + ')';
      statusEl.className = 'cache-status cached';
      if (dlBtn) { dlBtn.textContent = '✓ Cached'; dlBtn.disabled = true; }
    } else {
      statusEl.textContent = 'Not cached';
      statusEl.className = 'cache-status';
      if (dlBtn) { dlBtn.textContent = 'Download for offline use'; dlBtn.disabled = false; }
    }
  }

  /** Downloads the current model into the persistent cache without analyzing. */
  async function downloadForOffline() {
    var CM = FT.modelCache;
    if (!CM || !CM.available) return;

    var modelUrl = ($('modelUrl').value || '').trim() || 'https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx';
    var progBar = $('progressBar');
    var progFill = $('progressFill');
    if (progBar) progBar.classList.remove('hidden');

    try {
      await CM.downloadAndCache(modelUrl, function onProg(_a) {
        var pct = (_a && _a.percent) || 0;
        if (progFill) { progFill.style.width = pct + '%'; progFill.textContent = pct + '%'; }
      });
      setStatus('Model cached for offline use.');
      updateCacheStatus();
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      if (progBar) progBar.classList.add('hidden');
    }
  }

  function init() {
    if (!$('analyzeBtn')) return;
    $('analyzeBtn').addEventListener('click', analyze);
    $('logBtn').addEventListener('click', logEntry);
    $('portionGrams').addEventListener('input', updatePortion);
    // Cache controls
    var dlBtn = $('downloadBtn');
    if (dlBtn) dlBtn.addEventListener('click', downloadForOffline);
    var mu = $('modelUrl');
    if (mu) mu.addEventListener('input', updateCacheStatus);
    updateCacheStatus();
    // populate model selector
    const sel = $('modelId');
    if (sel) {
      Object.values(M.MODEL_SPECS).forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.label;
        sel.appendChild(opt);
      });
    }
    renderEntries();
  }

  FT.app = {
    loadModel, analyze, urlToDataUrl, fileToDataUrl,
    renderEntries, readStore, writeStore, updatePortion,
    updateCacheStatus, downloadForOffline,
    _getSession: () => session
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }
})(typeof window !== 'undefined' ? window : globalThis);
