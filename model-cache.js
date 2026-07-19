/*
 * Food Tracker - persistent browser storage for ONNX model files.
 *
 * Stores model binaries in IndexedDB so the ~93 MB model downloads once and
 * loads from local storage on every subsequent visit.  IndexedDB is used
 * instead of Cache API because Cache API rejects manually-constructed
 * Responses for cross-origin URLs (Hugging Face CDN redirects), whereas
 * IndexedDB stores raw ArrayBuffers with zero CORS friction.
 *
 * Part of the FT namespace; loaded before app.js.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});

  var DB_NAME = 'food-model-cache';
  var DB_VERSION = 1;
  var STORE_NAME = 'models';

  var _db = null;

  function idbAvailable() {
    try { return !!global.indexedDB; } catch (e) { return false; }
  }

  /** Opens (or creates) the IndexedDB database. Singleton – opened once. */
  function openDB() {
    if (_db) return Promise.resolve(_db);
    if (!idbAvailable()) return Promise.reject(new Error('IndexedDB not available'));

    return new Promise(function (resolve, reject) {
      var req = global.indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = function () {
        if (!req.result.objectStoreNames.contains(STORE_NAME)) {
          req.result.createObjectStore(STORE_NAME);
        }
      };
      req.onsuccess = function () {
        _db = req.result;
        resolve(_db);
      };
      req.onerror = function () {
        reject(req.error);
      };
    });
  }

  /** Returns true when the model at `url` has already been persisted. */
  async function isCached(url) {
    try {
      var db = await openDB();
      return new Promise(function (resolve) {
        var tx = db.transaction(STORE_NAME, 'readonly');
        var req = tx.objectStore(STORE_NAME).getKey(url);
        req.onsuccess = function () { resolve(!!req.result); };
        req.onerror = function () { resolve(false); };
      });
    } catch (e) { return false; }
  }

  /** Size of a stored model in bytes, or null. */
  async function getCachedSize(url) {
    try {
      var db = await openDB();
      return new Promise(function (resolve) {
        var tx = db.transaction(STORE_NAME, 'readonly');
        var req = tx.objectStore(STORE_NAME).get(url);
        req.onsuccess = function () {
          var buf = req.result;
          resolve(buf ? buf.byteLength : null);
        };
        req.onerror = function () { resolve(null); };
      });
    } catch (e) { return null; }
  }

  /** Loads a stored model, returning an ArrayBuffer or null. */
  async function loadFromCache(url) {
    try {
      var db = await openDB();
      return new Promise(function (resolve) {
        var tx = db.transaction(STORE_NAME, 'readonly');
        var req = tx.objectStore(STORE_NAME).get(url);
        req.onsuccess = function () {
          var buf = req.result;
          if (buf) {
            console.log('[model-cache] Loaded from IndexedDB:', url, '(' + (buf.byteLength / 1024 / 1024).toFixed(1) + ' MB)');
          }
          resolve(buf || null);
        };
        req.onerror = function () { resolve(null); };
      });
    } catch (e) { return null; }
  }

  /**
   * Downloads a model from the network, persists it in IndexedDB, and returns
   * its ArrayBuffer.
   *
   * @param {string} url - model URL to fetch
   * @param {function} [onProgress] - called with {loaded, total, percent} during download
   * @returns {Promise<ArrayBuffer>}
   */
  async function downloadAndCache(url, onProgress) {
    var response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch model: HTTP ' + response.status);

    var contentLength = response.headers.get('content-length');
    var total = contentLength ? parseInt(contentLength, 10) : 0;
    var arrayBuffer;

    if (response.body && total && onProgress) {
      // Stream with progress tracking.
      var reader = response.body.getReader();
      var chunks = [];
      var loaded = 0;

      while (true) {
        var step = await reader.read();
        if (step.done) break;
        chunks.push(step.value);
        loaded += step.value.length;
        onProgress({ loaded: loaded, total: total, percent: Math.round((loaded / total) * 100) });
      }

      var joined = new Uint8Array(loaded);
      var pos = 0;
      for (var i = 0; i < chunks.length; i++) {
        joined.set(chunks[i], pos);
        pos += chunks[i].length;
      }
      arrayBuffer = joined.buffer;
    } else {
      // No progress tracking — read the whole body at once.
      arrayBuffer = await response.arrayBuffer();
      if (onProgress) onProgress({ loaded: arrayBuffer.byteLength, total: arrayBuffer.byteLength, percent: 100 });
    }

    // Store the raw ArrayBuffer in IndexedDB (no CORS issues).
    try {
      var db = await openDB();
      await new Promise(function (resolve, reject) {
        var tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put(arrayBuffer, url);
        tx.oncomplete = function () { resolve(); };
        tx.onerror = function () { reject(tx.error); };
      });
      console.log('[model-cache] Stored in IndexedDB:', url, '(' + (arrayBuffer.byteLength / 1024 / 1024).toFixed(1) + ' MB)');
    } catch (e) {
      console.warn('[model-cache] IndexedDB store failed:', e.message || e);
    }

    return arrayBuffer;
  }

  /** Deletes a single stored model. */
  async function removeCached(url) {
    try {
      var db = await openDB();
      return new Promise(function (resolve) {
        var tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).delete(url);
        tx.oncomplete = function () { resolve(); };
        tx.onerror = function () { resolve(); };
      });
    } catch (e) { /* ignore */ }
  }

  /** Deletes all stored models. */
  async function clearAll() {
    try {
      var db = await openDB();
      return new Promise(function (resolve) {
        var tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
        tx.oncomplete = function () { resolve(); };
        tx.onerror = function () { resolve(); };
      });
    } catch (e) { /* ignore */ }
  }

  FT.modelCache = {
    available: idbAvailable(),
    isCached: isCached,
    getCachedSize: getCachedSize,
    loadFromCache: loadFromCache,
    downloadAndCache: downloadAndCache,
    removeCached: removeCached,
    clearAll: clearAll
  };
})(typeof window !== 'undefined' ? window : globalThis);
