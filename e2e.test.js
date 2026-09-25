/*
 * End-to-end contract test. It uses a fake ONNX Runtime so it exercises the
 * complete browser flow deterministically without downloading a 93 MB model.
 */
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { once } = require('node:events');
const puppeteer = require('puppeteer');

const root = __dirname;
const port = 0;
const fakeOrt = `
  window.__modelLoads = 0;
  window.ort = {
    Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } },
    InferenceSession: { create: async () => { window.__modelLoads++; return ({
      inputNames: ['input'], outputNames: ['logits'],
      run: async () => { const values = new Float32Array(101); values[31] = 20; return { logits: { data: values } }; }
    }); } }
  };
`;

function server() {
  return http.createServer((req, res) => {
    const requestPath = req.url === '/' ? '/index.html' : req.url;
    if (requestPath === '/ort.min.js') {
      res.writeHead(200, { 'content-type': 'application/javascript' });
      return res.end(fakeOrt);
    }
    const file = path.resolve(root, '.' + requestPath.split('?')[0]);
    if (!file.startsWith(root) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
    res.writeHead(200);
    res.end(fs.readFileSync(file));
  });
}

async function main() {
  const app = server();
  app.listen(port, '127.0.0.1');
  await once(app, 'listening');
  const address = app.address();
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 900, height: 900 });
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      if (request.url().includes('onnxruntime-web')) request.respond({ status: 200, contentType: 'application/javascript', body: fakeOrt });
      else request.continue();
    });
    await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'networkidle0' });

    // New contract: a detected food must expose an editable portion and update calories.
    assert.ok(await page.$('#portionGrams'), 'portion input is available');
    await page.$eval('#fileInput', (input) => {
      const svg = new Blob(['<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="pink"/></svg>'], { type: 'image/svg+xml' });
      const dt = new DataTransfer(); dt.items.add(new File([svg], 'meal.svg', { type: 'image/svg+xml' })); input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    // Choosing a photo analyzes it immediately, no extra click.
    await page.waitForSelector('#resultCard:not(.hidden)');
    assert.match(await page.$eval('#result-title', (el) => el.textContent), /Donuts/);
    assert.equal(await page.$eval('#portionGrams', (el) => el.value), '100');
    assert.equal(await page.$eval('#resultKcal', (el) => el.textContent), '452');
    const timing = await page.$eval('#status', (el) => Number((el.textContent.match(/(\d+) ms/) || [])[1]));
    assert.ok(Number.isFinite(timing) && timing < 1000, `mocked inference should stay responsive (was ${timing} ms)`);

    // Changing the model URL must create a new session, not reuse stale model state.
    await page.$eval('#modelUrl', (el) => { el.value = 'model-v2.onnx'; });
    await page.click('#analyzeBtn');
    await page.waitForFunction(() => window.__modelLoads === 2);
    assert.equal(await page.evaluate(() => window.__modelLoads), 2);

    await page.$eval('#portionGrams', (el) => { el.value = '50'; el.dispatchEvent(new Event('input', { bubbles: true })); });
    assert.equal(await page.$eval('#resultKcal', (el) => el.textContent), '226');
    await page.click('.step[data-step="10"]');
    assert.equal(await page.$eval('#portionGrams', (el) => el.value), '60');
    await page.click('.step[data-step="-10"]');
    assert.equal(await page.$eval('#resultKcal', (el) => el.textContent), '226');
    await page.click('#logBtn');
    assert.match(await page.$eval('#entries', (el) => el.textContent), /Donuts[\s\S]*226 kcal/);
    assert.match(await page.$eval('#totalCalories', (el) => el.textContent), /226 kcal/);
    assert.ok(await page.$('#resultCard.hidden'), 'result closes after logging');
    assert.ok(await page.$('#stagePreview.hidden'), 'stage resets after logging');

    // Camera: capture a frame from Chromium's fake device and analyze it.
    await page.click('#cameraBtn');
    await page.waitForSelector('#shutterBtn:not([disabled])');
    await page.click('#shutterBtn');
    await page.waitForSelector('#camera.hidden');
    await page.waitForSelector('#resultCard:not(.hidden)');
    assert.equal(await page.evaluate(() => document.querySelector('#cameraVideo').srcObject), null, 'camera stream is released');
    await page.click('#logBtn');
    assert.match(await page.$eval('#totalCalories', (el) => el.textContent), /678 kcal/);

    // Entries can be removed.
    await page.click('#entries .entry-del');
    assert.match(await page.$eval('#totalCalories', (el) => el.textContent), /226 kcal/);

    // The browser unit suite is also part of the automated gate.
    await page.goto(`http://127.0.0.1:${address.port}/test.html`, { waitUntil: 'networkidle0' });
    await page.click('#runBtn');
    await page.waitForFunction(() => /^\d+\/\d+ passed$/.test(document.querySelector('#summary').textContent));
    const summary = await page.$eval('#summary', (el) => el.textContent);
    const failures = await page.$$('.fail');
    const failureText = await page.$$eval('.fail', (els) => els.map((el) => el.textContent).join('\n'));
    assert.equal(failures.length, 0, `browser unit test failures: ${summary}\n${failureText}`);
  } finally {
    await browser.close();
    app.close();
  }
}

main().then(() => console.log('E2E PASS')).catch((error) => { console.error(error); process.exitCode = 1; });
