/*
 * Food Tracker - persistent browser caching for ONNX model files.
 *
 * Uses the Cache API (window.caches) to store model binaries so the ~93 MB
 * model downloads once and loads from cache on subsequent visits. Falls back
 * gracefully when the Cache API is unavailable (HTTP, older browsers).
 *
 * Part of the FT namespace; loaded before app.js.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});

  const CACHE_NAME = 'food-model-v1';

  function cacheAvailable() {
    try { return !!global.caches; } catch (e) { return false; }
  }

  async function openCache() {
    if (!cacheAvailable()) throw new Error('Cache API not available');
    return global.caches.open(CACHE_NAME);
  }

  /** Returns true when the model at `url` has already been cached. */
  async function isCached(url) {
    try {
      const cache = await openCache();
      return !!(await cache.match(url));
    } catch (e) { return false; }
  }

  /** Size of a cached entry in bytes, or null if not cached / unavailable. */
  async function getCachedSize(url) {
    try {
      const cache = await openCache();
      const match = await cache.match(url);
      if (!match) return null;
      const blob = await match.blob();
      return blob.size;
    } catch (e) { return null; }
  }

  /** Loads a model from cache, returning an ArrayBuffer or null. */
  async function loadFromCache(url) {
    try {
      const cache = await openCache();
      const match = await cache.match(url);
      if (!match) return null;
      return match.arrayBuffer();
    } catch (e) { return null; }
  }

  /**
   * Downloads a model from the network, stores it in the Cache API, and returns
   * its ArrayBuffer.
   *
   * @param {string} url - model URL to fetch
   * @param {function} [onProgress] - called with {loaded, total, percent} during download
   * @returns {Promise<ArrayBuffer>}
   */
  async function downloadAndCache(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch model: HTTP ' + response.status);

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    // When we can't stream (no body or no length), buffer and cache in one go.
    if (!response.body || !total) {
      const buffer = await response.arrayBuffer();
      try {
        const cache = await openCache();
        await cache.put(url, new Response(buffer, { headers: response.headers }));
      } catch (e) { /* cache write failed – caller still has the buffer */ }
      if (onProgress) onProgress({ loaded: buffer.byteLength, total: buffer.byteLength, percent: 100 });
      return buffer;
    }

    // Stream the response: track progress, assemble chunks, then cache.
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (onProgress) {
        onProgress({ loaded, total, percent: Math.round((loaded / total) * 100) });
      }
    }

    // Combine chunks into one ArrayBuffer.
    const buffer = new Uint8Array(loaded);
    let pos = 0;
    for (let i = 0; i < chunks.length; i++) {
      buffer.set(chunks[i], pos);
      pos += chunks[i].length;
    }
    const arrayBuffer = buffer.buffer;

    // Store in cache (best-effort – caller already has the data).
    try {
      const cache = await openCache();
      await cache.put(url, new Response(arrayBuffer, { headers: response.headers }));
    } catch (e) { /* non-critical */ }

    return arrayBuffer;
  }

  /** Deletes a single cached model entry. */
  async function removeCached(url) {
    try {
      const cache = await openCache();
      await cache.delete(url);
    } catch (e) { /* ignore */ }
  }

  /** Deletes the entire model cache (all models). */
  async function clearAll() {
    try {
      await global.caches.delete(CACHE_NAME);
    } catch (e) { /* ignore */ }
  }

  FT.modelCache = {
    available: cacheAvailable(),
    isCached,
    getCachedSize,
    loadFromCache,
    downloadAndCache,
    removeCached,
    clearAll
  };
})(typeof window !== 'undefined' ? window : globalThis);
