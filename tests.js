/*
 * Food Tracker tests - pure logic + mocked network, run in the browser.
 * Open test.html and click "Run tests". No Node, no npm, no build step.
 */
(function (global) {
  const FT = global.FT;
  const { test, assert } = FT.test;
  const M = FT.models;

  test('validateImageUrl accepts http(s) URLs', () => {
    assert.equal(M.validateImageUrl('https://x.com/a.png'), 'https://x.com/a.png');
    assert.equal(M.validateImageUrl('http://x.com/a.png'), 'http://x.com/a.png');
  });

  test('validateImageUrl rejects non-http(s) and invalid input', () => {
    assert.throws(() => M.validateImageUrl('ftp://x.com/a.png'));
    assert.throws(() => M.validateImageUrl('not a url'));
    assert.throws(() => M.validateImageUrl(''));
  });

  test('imageToTensorChannels lays out CHW with [1,3,H,W] dims', () => {
    // 2x2 image -> 4 pixels * 4 channels = 16 bytes; 3 channels * 4 = 12 floats
    const pixels = new Uint8ClampedArray([
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 100, 100, 100, 255
    ]);
    const { data, dims } = M.imageToTensorChannels(pixels, 2, [0, 0, 0], [1, 1, 1]);
    assert.equal(dims[0], 1);
    assert.equal(dims[1], 3);
    assert.equal(dims[2], 2);
    assert.equal(dims[3], 2);
    assert.equal(data.length, 12);
    // R plane first, then G, then B
    assert.equal(data[0], 255 / 255); // top-left red
    assert.equal(data[4], 0 / 255);   // top-left green
    assert.equal(data[8], 0 / 255);   // top-left blue
  });

  test('imageToTensorChannels applies mean/std normalization', () => {
    const pixels = new Uint8ClampedArray([100, 150, 200, 255]); // single pixel
    const { data } = M.imageToTensorChannels(
      pixels, 1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    );
    assert.near(data[0], (100 / 255 - 0.5) / 0.5, 1e-7);
    assert.near(data[1], (150 / 255 - 0.5) / 0.5, 1e-7);
    assert.near(data[2], (200 / 255 - 0.5) / 0.5, 1e-7);
  });

  test('postprocess maps raw outputs to nutrition (NutritionVerse-Direct order)', () => {
    const spec = M.MODEL_SPECS['nutritionverse-direct'];
    const out = M.postprocess([250.4, 300.6, 20.1, 10.2, 30.7], spec);
    assert.equal(out.calories, 250);          // rounded to int
    assert.equal(out.mass, 300.6);            // 1 decimal
    assert.equal(out.protein, 20.1);
    assert.equal(out.fat, 10.2);
    assert.equal(out.carbs, 30.7);
    assert.equal(out.caloriesUnit, 'kcal');
  });

  test('postprocess clamps negative predictions to 0', () => {
    const spec = M.MODEL_SPECS['foodcnn-nutrition5k'];
    const out = M.postprocess([-5, -1, 12.3, 4.4, 0], spec);
    assert.equal(out.calories, 0);
    assert.equal(out.mass, 0);
    assert.equal(out.fat, 12.3);
    assert.equal(out.carbs, 4.4);
    assert.equal(out.protein, 0);
  });

  test('postprocess handles separate named outputs', () => {
    const spec = { outputs: [
      { name: 'c', key: 'calories', unit: 'kcal' },
      { name: 'm', key: 'mass', unit: 'g' }
    ] };
    const out = M.postprocess({ c: 123.9, m: 45.5 }, spec);
    assert.equal(out.calories, 124);
    assert.equal(out.mass, 45.5);
  });

  test('getSpec falls back to the bundled default for unknown ids', () => {
    assert.equal(M.getSpec('does-not-exist').id, 'swin-food101');
    assert.equal(M.getSpec('foodcnn-nutrition5k').id, 'foodcnn-nutrition5k');
  });

  test('softmax produces a normalized distribution', () => {
    const p = M.softmax([1, 2, 3]);
    const sum = p.reduce((s, x) => s + x, 0);
    assert.ok(Math.abs(sum - 1) < 1e-6);
    assert.ok(p[2] > p[1] && p[1] > p[0]);
  });

  test('argmax returns the index of the largest value', () => {
    assert.equal(M.argmax([0.1, 0.7, 0.2]), 1);
    assert.equal(M.argmax([5, 1, 2, 9, 3]), 3);
  });

  test('classifier postprocess maps logits to the predicted dish nutrition', () => {
    const spec = M.MODEL_SPECS['swin-food101'];
    // 101 logits, all zero except "donuts" (index 31) set high.
    const logits = new Array(101).fill(0);
    const donutIdx = FT.nutrition.LABELS.indexOf('donuts');
    logits[donutIdx] = 20;
    const out = M.postprocess(logits, spec);
    assert.equal(out.label, 'donuts');
    assert.equal(out.name, 'Donuts');
    assert.equal(out.classIndex, donutIdx);
    assert.ok(out.confidence > 99);
    const expected = FT.nutrition.NUTRITION['donuts'];
    assert.equal(out.calories, Math.round(expected[0]));
    assert.equal(out.protein, expected[2]);
  });

  test('nutrition scales a recognised food to the selected portion', () => {
    const half = FT.nutrition.nutritionForLabelAndMass('donuts', 50);
    assert.equal(half.mass, 50);
    assert.equal(half.calories, 226);
    assert.equal(half.protein, 2.45);
    assert.throws(() => FT.nutrition.nutritionForLabelAndMass('donuts', 0));
  });

  test('urlToDataUrl fetches and converts a remote image', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, blob: async () => ({}) });
    const origFR = global.FileReader;
    global.FileReader = class {
      readAsDataURL() { this.result = 'data:image/png;base64,REMOTE'; this.onload(); }
    };
    try {
      const out = await FT.app.urlToDataUrl('https://x.com/food.png');
      assert.equal(out, 'data:image/png;base64,REMOTE');
    } finally {
      global.fetch = origFetch;
      global.FileReader = origFR;
    }
  });

  test('urlToDataUrl rejects a failed fetch', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => ({ ok: false, status: 404 });
    try {
      await FT.app.urlToDataUrl('https://x.com/missing.png');
      assert.ok(false, 'should have thrown');
    } catch (e) {
      assert.ok(/404/.test(e.message));
    } finally {
      global.fetch = origFetch;
    }
  });

  test('loadModel creates an ORT session from its cached global', async () => {
    const realOrt = global.ort;
    let createdWith = null;
    global.ort = {
      InferenceSession: {
        create: async (url, opts) => {
          createdWith = url;
          return { inputNames: ['input'], outputNames: ['output'], run: async () => ({}) };
        }
      }
    };
    try {
      const url = `unit-model-${Date.now()}.onnx`;
      const session = await FT.app.loadModel(url, 'swin-food101');
      assert.ok(session && typeof session.run === 'function');
      assert.equal(createdWith, url);
    } finally {
      global.ort = realOrt;
    }
  });

  // --- catalog integrity tests ---

  test('all 101 LABELS have nutrition entries', () => {
    const N = FT.nutrition;
    assert.equal(N.LABELS.length, 101);
    let missing = 0;
    N.LABELS.forEach(function (label) {
      if (!N.NUTRITION[label]) missing++;
    });
    assert.equal(missing, 0, missing + ' labels missing from NUTRITION table');
  });

  test('every nutrition entry has positive calories', () => {
    const N = FT.nutrition;
    var zeroCal = [];
    Object.keys(N.NUTRITION).forEach(function (label) {
      var v = N.NUTRITION[label];
      if (!v || v[0] <= 0) zeroCal.push(label);
    });
    assert.equal(zeroCal.length, 0, 'Zero-calorie foods: ' + zeroCal.join(', '));
  });

  test('nutritionForLabel and nutritionForIndex agree for all 101 indices', () => {
    const N = FT.nutrition;
    for (var i = 0; i < 101; i++) {
      var byIdx = N.nutritionForIndex(i);
      var byLabel = N.nutritionForLabel(N.LABELS[i]);
      assert.equal(byIdx.label, byLabel.label);
      assert.equal(byIdx.calories, byLabel.calories);
      assert.equal(byIdx.mass, byLabel.mass);
      assert.equal(byIdx.protein, byLabel.protein);
      assert.equal(byIdx.fat, byLabel.fat);
      assert.equal(byIdx.carbs, byLabel.carbs);
    }
  });

  test('nutritionForLabelAndMass scales per-100g correctly', () => {
    const N = FT.nutrition;
    // Pizza: 266 kcal/100g
    var r = N.nutritionForLabelAndMass('pizza', 150);
    assert.equal(r.mass, 150);
    assert.equal(r.calories, 266 * 1.5);   // 399
    assert.equal(r.protein, 11 * 1.5);      // 16.5
    assert.equal(r.fat, 9.7 * 1.5);         // 14.55
    assert.equal(r.carbs, 33.3 * 1.5);      // 49.95
  });

  test('nutritionForLabelAndMass handles fractional grams', () => {
    const N = FT.nutrition;
    var r = N.nutritionForLabelAndMass('steak', 75.5);
    assert.equal(r.mass, 75.5);
    assert.equal(r.calories, 271 * 0.755);  // ~204.605
    assert.equal(r.protein, 25 * 0.755);    // 18.875
  });

  test('nutritionForLabelAndMass rejects invalid portions', () => {
    var N = FT.nutrition;
    assert.throws(function () { N.nutritionForLabelAndMass('pizza', 0); });
    assert.throws(function () { N.nutritionForLabelAndMass('pizza', -5); });
    assert.throws(function () { N.nutritionForLabelAndMass('pizza', NaN); });
    assert.throws(function () { N.nutritionForLabelAndMass('pizza', Infinity); });
  });

  test('nutritionForLabelAndMass works for unknown label (falls back to zeros)', function () {
    var N = FT.nutrition;
    // Unknown label: calories=0, mass=100 — scaling 200g should give 0 cals, 200g mass
    var r = N.nutritionForLabelAndMass('nonexistent_food', 200);
    assert.equal(r.mass, 200);
    assert.equal(r.calories, 0);
    assert.equal(r.protein, 0);
    assert.equal(r.fat, 0);
    assert.equal(r.carbs, 0);
  });

  test('CATALOG_VERSION is defined and follows semver', function () {
    var v = FT.nutrition.CATALOG_VERSION;
    assert.ok(typeof v === 'string' && v.length > 0, 'CATALOG_VERSION must be a non-empty string');
    assert.ok(/^\d+\.\d+\.\d+/.test(v), 'CATALOG_VERSION should be semver-like, got: ' + v);
  });

  test('prettyLabel formats underscore labels', function () {
    var N = FT.nutrition;
    assert.equal(N.prettyLabel('chicken_curry'), 'Chicken Curry');
    assert.equal(N.prettyLabel('macaroni_and_cheese'), 'Macaroni And Cheese');
    assert.equal(N.prettyLabel('donuts'), 'Donuts');
    assert.equal(N.prettyLabel(''), '');
  });

  // --- preprocessing parity test (single-pixel golden value) ---
  // This tensor is the expected output for a 1×1 red pixel [255,0,0,255]
  // preprocessed with ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
  // The Python reference produces: R=(1-0.485)/0.229=2.2489, G=(0-0.456)/0.224=-2.0357, B=(0-0.406)/0.225=-1.8044
  test('imageToTensorChannels produces correct normalized values (parity with Python)', function () {
    var pixels = new Uint8ClampedArray([255, 0, 0, 255]); // single red pixel
    var mean = [0.485, 0.456, 0.406];
    var std  = [0.229, 0.224, 0.225];
    var result = M.imageToTensorChannels(pixels, 1, mean, std);
    assert.equal(result.dims[0], 1);
    assert.equal(result.dims[1], 3);
    assert.equal(result.dims[2], 1);
    assert.equal(result.dims[3], 1);
    // R channel: (255/255 - 0.485) / 0.229 = (1 - 0.485) / 0.229
    assert.near(result.data[0], (1 - 0.485) / 0.229, 1e-4);
    // G channel: (0/255 - 0.456) / 0.224 = -0.456 / 0.224
    assert.near(result.data[1], (0 - 0.456) / 0.224, 1e-4);
    // B channel: (0/255 - 0.406) / 0.225 = -0.406 / 0.225
    assert.near(result.data[2], (0 - 0.406) / 0.225, 1e-4);
  });

  // --- regression postprocess with real model output shapes ---
  test('postprocess regression handles 5-element array (NV-Direct order)', function () {
    var spec = M.MODEL_SPECS['nutritionverse-direct'];
    // Simulate a real model output: ~250 kcal, ~300g, 20g protein, 10g fat, 30g carbs
    var out = M.postprocess([250.4, 300.6, 20.1, 10.2, 30.7], spec);
    assert.equal(out.calories, 250);     // rounded to int
    assert.equal(out.mass, 300.6);       // 1 decimal
    assert.equal(out.protein, 20.1);
    assert.equal(out.fat, 10.2);
    assert.equal(out.carbs, 30.7);
    assert.equal(out.caloriesUnit, 'kcal');
    assert.equal(out.massUnit, 'g');
  });

  test('postprocess regression clamps negative predictions to 0', function () {
    var spec = M.MODEL_SPECS['foodcnn-nutrition5k'];
    var out = M.postprocess([-5, -1, 12.3, 4.4, 0], spec);
    assert.equal(out.calories, 0);
    assert.equal(out.mass, 0);
    assert.equal(out.fat, 12.3);
    assert.equal(out.carbs, 4.4);
    assert.equal(out.protein, 0);
  });

  // --- model spec validation ---
  test('all model specs have required fields', function () {
    var ids = ['swin-food101', 'nutritionverse-direct', 'foodcnn-nutrition5k'];
    ids.forEach(function (id) {
      var spec = M.MODEL_SPECS[id];
      assert.ok(spec, 'Missing spec: ' + id);
      assert.ok(typeof spec.size === 'number' && spec.size > 0, id + ': invalid size');
      assert.ok(Array.isArray(spec.mean) && spec.mean.length === 3, id + ': invalid mean');
      assert.ok(Array.isArray(spec.std) && spec.std.length === 3, id + ': invalid std');
      assert.ok(spec.type === 'classifier' || spec.type === 'regressor', id + ': invalid type');
    });
  });

  test('swin-food101 spec has numClasses=101', function () {
    assert.equal(M.MODEL_SPECS['swin-food101'].numClasses, 101);
  });

  if (typeof document !== 'undefined' && document.getElementById('results')) {
    FT.test.run().then(function (r) { FT.test.render(r, document.getElementById('results')); });
  }
})(typeof window !== 'undefined' ? window : globalThis);
