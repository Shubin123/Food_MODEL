/*
 * Food Tracker - UI + ONNX Runtime Web inference (no frameworks, no build step).
 * Loads a converted nutrition model (.onnx) and runs it entirely client-side.
 * Depends on: models.js (FT.models) and the global `ort` from the CDN script.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});
  const M = FT.models;
  const STORE_KEY = 'food-tracker-entries';
  const DEFAULT_MODEL_URL = 'https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx';

  let session = null;
  let activeSpec = null;
  let sessionKey = null;

  // In-flight model load, shared so warm-up and analysis never load twice.
  let loading = null;
  let loadingKey = null;

  // Current photo: { blob } from upload/camera/drop/paste, or { url }.
  let source = null;
  let previewUrl = null;
  let runId = 0;

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

  function currentModelUrl() {
    return ($('modelUrl').value || '').trim() || DEFAULT_MODEL_URL;
  }

  function currentSpecId() {
    const sel = $('modelId');
    return sel && sel.value ? sel.value : 'swin-food101';
  }

  function onModelProgress(p) {
    const pct = (p && p.percent) || 0;
    $('progressFill').style.width = pct + '%';
    $('progressFill').textContent = pct + '%';
  }

  // Loads the selected model, sharing any load already in flight.
  async function ensureModel() {
    const modelUrl = currentModelUrl();
    const specId = currentSpecId();
    const key = `${modelUrl}|${specId}`;
    if (session && sessionKey === key) return session;
    if (loading && loadingKey === key) return loading;

    const CM = FT.modelCache;
    const wasCached = CM && CM.available && await CM.isCached(modelUrl).catch(() => false);
    if (!wasCached) $('progressBar').classList.remove('hidden');
    loadingKey = key;
    loading = loadModel(modelUrl, specId, onModelProgress).finally(() => {
      if (loadingKey === key) { loading = null; loadingKey = null; }
      $('progressBar').classList.add('hidden');
      updateCacheStatus();
    });
    return loading;
  }

  // Starts loading the model in the background (e.g. while the camera is open).
  function warmModel() {
    if (!global.ort) return;
    ensureModel().catch(() => { /* surfaced again on analyze */ });
  }

  async function analyze() {
    const run = ++runId;
    const stage = $('stage');
    $('resultCard').classList.add('hidden');

    // Programmatically populated file inputs don't fire `change`.
    const file = $('fileInput').files && $('fileInput').files[0];
    if (!source && file) setPreview({ blob: file });
    const url = $('imageUrl').value.trim();
    if (!source && url) setPreview({ url });
    if (!source) return setStatus('Take or choose a photo first.', true);

    let imageEl;
    setStatus('Reading photo…');
    stage.classList.add('busy');
    try {
      if (source.blob) {
        imageEl = await loadImage(previewUrl);
      } else {
        imageEl = await loadImage(await urlToDataUrl(source.url));
      }
    } catch (err) {
      stage.classList.remove('busy');
      return setStatus(err.message, true);
    }

    try {
      if (!session || sessionKey !== `${currentModelUrl()}|${currentSpecId()}`) {
        setStatus('Loading model…');
        await ensureModel();
      }
      if (run !== runId) return;
      setStatus('Analyzing…');
      const pre = M.preprocessImage(imageEl, activeSpec);
      const tensor = new global.ort.Tensor('float32', pre.data, pre.dims);
      const feeds = {};
      feeds[session.inputNames[0]] = tensor;

      const t0 = performance.now();
      const results = await session.run(feeds);
      const dt = Math.round(performance.now() - t0);
      if (run !== runId) return;

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
      nutrition.thumb = makeThumb(imageEl);
      renderResult(nutrition);
      setStatus(`Ran locally in ${dt} ms · ${activeSpec.label}`);
    } catch (err) {
      if (run === runId) setStatus(err.message, true);
    } finally {
      if (run === runId) stage.classList.remove('busy');
    }
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

  // Small square JPEG kept with the log entry.
  function makeThumb(img) {
    try {
      const size = 96;
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      const s = Math.min(w, h);
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      canvas.getContext('2d').drawImage(img, (w - s) / 2, (h - s) / 2, s, s, 0, 0, size, size);
      return canvas.toDataURL('image/jpeg', 0.7);
    } catch (e) {
      return null;
    }
  }

  const MACROS = [
    ['protein', 'Protein', 'var(--protein)', 4],
    ['fat', 'Fat', 'var(--fat)', 9],
    ['carbs', 'Carbs', 'var(--carbs)', 4]
  ];

  function renderResult(n, keepInput) {
    $('resultCard').classList.remove('hidden');
    $('result-title').textContent = n.name || 'Estimated meal';
    $('resultConfidence').textContent = typeof n.confidence === 'number'
      ? `${n.confidence}% match · ${n.mass} g` : `${n.mass} g`;
    $('resultKcal').textContent = String(n.calories);

    const body = $('resultBody');
    body.innerHTML = '';
    const bar = document.createElement('div');
    bar.className = 'macro-bar';
    MACROS.forEach(([key, label, color, kcalPerGram]) => {
      const cell = document.createElement('div');
      cell.className = 'macro';
      cell.style.setProperty('--dot', color);
      cell.innerHTML = '<div class="macro-label"></div><div class="macro-value"></div>';
      cell.children[0].textContent = label;
      cell.children[1].textContent = `${n[key]} g`;
      body.appendChild(cell);

      const seg = document.createElement('span');
      seg.style.setProperty('--dot', color);
      seg.style.flexGrow = String(Math.max(0, n[key] * kcalPerGram));
      bar.appendChild(seg);
    });
    body.appendChild(bar);

    FT._lastResult = n;
    const editor = $('portionEditor');
    if (n.mass > 0) {
      if (!keepInput) $('portionGrams').value = String(Math.round(n.mass));
      editor.classList.remove('hidden');
      $('estimateNote').textContent = n.label
        ? 'Scaled from the detected food’s reference nutrition. Check the grams before logging.'
        : 'Model estimate. Adjusting grams scales it proportionally.';
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
    if (!Number.isFinite(grams) || grams <= 0) return;
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
    renderResult(current, true);
  }

  function stepPortion(delta) {
    const input = $('portionGrams');
    const value = Math.max(1, Math.round((Number(input.value) || 0) + delta));
    input.value = String(value);
    updatePortion();
  }

  /* ---------- Photo sources ---------- */

  function setPreview(next) {
    if (previewUrl && previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    source = next;
    previewUrl = next.blob ? URL.createObjectURL(next.blob) : next.url;
    $('previewImg').src = previewUrl;
    $('stageEmpty').classList.add('hidden');
    $('stagePreview').classList.remove('hidden');
  }

  function usePhoto(next) {
    if (next.blob && next.blob.type && !next.blob.type.startsWith('image/')) {
      return setStatus('That file isn’t an image.', true);
    }
    setPreview(next);
    analyze();
  }

  function resetStage() {
    runId++;
    if (previewUrl && previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    source = null;
    previewUrl = null;
    FT._lastResult = null;
    $('previewImg').removeAttribute('src');
    $('stagePreview').classList.add('hidden');
    $('stageEmpty').classList.remove('hidden');
    $('stage').classList.remove('busy');
    $('resultCard').classList.add('hidden');
    $('fileInput').value = '';
    $('cameraInput').value = '';
    $('imageUrl').value = '';
  }

  function bindDropAndPaste() {
    const stage = $('stage');
    let depth = 0;
    stage.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; stage.classList.add('dragging'); });
    stage.addEventListener('dragover', (e) => e.preventDefault());
    stage.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; stage.classList.remove('dragging'); } });
    stage.addEventListener('drop', (e) => {
      e.preventDefault();
      depth = 0;
      stage.classList.remove('dragging');
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (f) usePhoto({ blob: f });
    });
    document.addEventListener('paste', (e) => {
      if (e.target.closest && e.target.closest('input, textarea')) return;
      const item = Array.from((e.clipboardData && e.clipboardData.items) || []).find((i) => i.type.startsWith('image/'));
      if (item) usePhoto({ blob: item.getAsFile() });
    });
  }

  /* ---------- Camera ---------- */

  let stream = null;
  let facing = 'environment';

  function hasCameraApi() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async function openCamera() {
    if (!hasCameraApi()) { $('cameraInput').click(); return; }
    $('camera').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    warmModel();
    await startStream();
  }

  async function startStream() {
    stopStream();
    const msg = $('cameraMsg');
    const shutter = $('shutterBtn');
    msg.classList.add('hidden');
    shutter.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1440 } },
        audio: false
      });
      if ($('camera').classList.contains('hidden')) return stopStream(); // closed while waiting
      const video = $('cameraVideo');
      video.srcObject = stream;
      await video.play().catch(() => {});
      const settings = stream.getVideoTracks()[0].getSettings();
      $('camera').classList.toggle('mirrored', (settings.facingMode || facing) === 'user');
      shutter.disabled = false;
      const devices = await navigator.mediaDevices.enumerateDevices();
      $('cameraFlip').classList.toggle('off', devices.filter((d) => d.kind === 'videoinput').length < 2);
    } catch (err) {
      const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
      msg.innerHTML = '';
      msg.append(denied ? 'Camera access was blocked.' : 'No camera is available.', document.createElement('br'));
      const fallback = document.createElement('button');
      fallback.type = 'button';
      fallback.className = 'btn ghost small';
      fallback.style.cssText = 'margin-top:12px;color:#fff;border-color:rgb(255 255 255 / 40%)';
      fallback.textContent = 'Choose a photo instead';
      fallback.addEventListener('click', () => { closeCamera(); $('fileInput').click(); });
      msg.append(fallback);
      msg.classList.remove('hidden');
    }
  }

  function stopStream() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    $('cameraVideo').srcObject = null;
  }

  function closeCamera() {
    stopStream();
    $('camera').classList.add('hidden');
    document.body.style.overflow = '';
  }

  function capture() {
    const video = $('cameraVideo');
    if (!stream || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const cam = $('camera');
    cam.classList.remove('flash');
    void cam.offsetWidth;
    cam.classList.add('flash');
    canvas.toBlob((blob) => {
      closeCamera();
      if (blob) usePhoto({ blob: new File([blob], 'camera.jpg', { type: 'image/jpeg' }) });
    }, 'image/jpeg', 0.92);
  }

  function flipCamera() {
    facing = facing === 'environment' ? 'user' : 'environment';
    startStream();
  }

  /* ---------- Log ---------- */

  function logEntry() {
    if (!FT._lastResult) return;
    const n = FT._lastResult;
    const entries = readStore();
    entries.push({
      calories: n.calories, mass: n.mass, protein: n.protein,
      fat: n.fat, carbs: n.carbs, name: n.name || null,
      model: n.model, thumb: n.thumb || null, at: new Date().toISOString()
    });
    try { writeStore(entries); }
    catch (e) {
      // Storage full: keep the entry, drop its thumbnail.
      entries[entries.length - 1].thumb = null;
      writeStore(entries);
    }
    resetStage();
    renderEntries();
    setStatus(`Added ${n.name || 'meal'} · ${n.calories} kcal`);
  }

  function deleteEntry(index) {
    const entries = readStore();
    entries.splice(index, 1);
    writeStore(entries);
    renderEntries();
  }

  function isToday(iso) {
    const d = new Date(iso);
    const now = new Date();
    return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
  }

  function renderEntries() {
    const today = readStore().map((e, i) => ({ e, i })).filter(({ e }) => isToday(e.at));
    const sum = (key) => Math.round(today.reduce((s, { e }) => s + (Number(e[key]) || 0), 0));
    $('totalCalories').textContent = `${sum('calories')} kcal`;
    $('totalMacros').textContent = `P ${sum('protein')} · F ${sum('fat')} · C ${sum('carbs')} g`;
    $('emptyLog').classList.toggle('hidden', today.length > 0);

    const ul = $('entries');
    ul.innerHTML = '';
    today.reverse().forEach(({ e, i }) => {
      const li = document.createElement('li');
      li.className = 'entry';
      li.innerHTML =
        '<img class="entry-thumb" alt="" />' +
        '<div class="entry-main"><div class="entry-name"></div><div class="entry-meta"></div></div>' +
        '<span class="entry-kcal"></span>' +
        '<button class="entry-del" type="button"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg></button>';
      const thumb = li.querySelector('.entry-thumb');
      if (e.thumb) thumb.src = e.thumb; else thumb.style.visibility = 'hidden';
      const name = e.name || 'Meal';
      const time = new Date(e.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      li.querySelector('.entry-name').textContent = name;
      li.querySelector('.entry-meta').textContent = `${time} · ${e.mass} g · P ${e.protein} · F ${e.fat} · C ${e.carbs}`;
      li.querySelector('.entry-kcal').textContent = `${e.calories} kcal`;
      const del = li.querySelector('.entry-del');
      del.setAttribute('aria-label', `Remove ${name}`);
      del.addEventListener('click', () => deleteEntry(i));
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

    var modelUrl = currentModelUrl();
    var cached = false;
    try { cached = await CM.isCached(modelUrl); } catch (e) { /* ignore */ }

    if (cached) {
      var sizeBytes = null;
      try { sizeBytes = await CM.getCachedSize(modelUrl); } catch (e) { /* ignore */ }
      var sizeStr = sizeBytes ? (sizeBytes / (1024 * 1024)).toFixed(1) + ' MB' : 'cached';
      statusEl.textContent = '✓ Available offline (' + sizeStr + ')';
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

    var progBar = $('progressBar');
    progBar.classList.remove('hidden');
    try {
      await CM.downloadAndCache(currentModelUrl(), onModelProgress);
      setStatus('Model cached for offline use.');
      updateCacheStatus();
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      progBar.classList.add('hidden');
    }
  }

  function init() {
    if (!$('analyzeBtn')) return;
    $('analyzeBtn').addEventListener('click', analyze);
    $('retakeBtn').addEventListener('click', () => { resetStage(); setStatus(''); });
    $('logBtn').addEventListener('click', logEntry);
    $('portionGrams').addEventListener('input', updatePortion);
    document.querySelectorAll('.step').forEach((b) => {
      b.addEventListener('click', () => stepPortion(Number(b.dataset.step)));
    });

    // Photo sources
    $('fileInput').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) usePhoto({ blob: f }); });
    $('cameraInput').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) usePhoto({ blob: f }); });
    $('uploadBtn').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); }
    });
    $('uploadBtn').tabIndex = 0;
    $('urlBtn').addEventListener('click', () => {
      const url = $('imageUrl').value.trim();
      if (url) usePhoto({ url });
    });
    bindDropAndPaste();

    // Camera
    $('cameraBtn').addEventListener('click', openCamera);
    $('cameraClose').addEventListener('click', closeCamera);
    $('shutterBtn').addEventListener('click', capture);
    $('cameraFlip').addEventListener('click', flipCamera);
    document.addEventListener('keydown', (e) => {
      if ($('camera').classList.contains('hidden')) return;
      if (e.key === 'Escape') closeCamera();
      if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); capture(); }
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && stream) closeCamera();
    });

    // Cache controls
    $('downloadBtn').addEventListener('click', downloadForOffline);
    $('modelUrl').addEventListener('input', updateCacheStatus);
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
    updateCacheStatus, downloadForOffline, openCamera, closeCamera,
    _getSession: () => session
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }
})(typeof window !== 'undefined' ? window : globalThis);
