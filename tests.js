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
    assert.equal(half.protein, 2.5);
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

  if (typeof document !== 'undefined' && document.getElementById('results')) {
    FT.test.run().then((r) => FT.test.render(r, document.getElementById('results')));
  }
})(typeof window !== 'undefined' ? window : globalThis);
