/*
 * Tiny browser test runner (no frameworks, no build step).
 * Run by opening test.html. Exposes FT.test for tests.js to register cases.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});

  const cases = [];
  function test(name, fn) {
    cases.push({ name, fn });
  }

  function assert(cond, msg) {
    if (!cond) throw new Error(msg || 'Assertion failed');
  }
  assert.equal = (a, b, msg) => {
    if (a !== b) throw new Error(msg || `Expected ${JSON.stringify(a)} === ${JSON.stringify(b)}`);
  };
  assert.ok = (a, msg) => assert(a, msg);
  assert.near = (actual, expected, epsilon, msg) => {
    if (Math.abs(actual - expected) > epsilon) {
      throw new Error(msg || `Expected ${actual} to be within ${epsilon} of ${expected}`);
    }
  };
  assert.throws = (fn, msg) => {
    let threw = false;
    try { fn(); } catch (e) { threw = true; }
    assert(threw, msg || 'Expected function to throw');
  };

  async function run() {
    const results = [];
    for (const c of cases) {
      try {
        await c.fn();
        results.push({ name: c.name, pass: true });
      } catch (err) {
        results.push({ name: c.name, pass: false, error: err.message });
      }
    }
    return results;
  }

  function render(results, el) {
    el.innerHTML = '';
    let passed = 0;
    results.forEach((r) => {
      if (r.pass) passed++;
      const li = document.createElement('li');
      li.className = r.pass ? 'pass' : 'fail';
      li.textContent = (r.pass ? 'PASS  ' : 'FAIL  ') + r.name + (r.pass ? '' : '  -> ' + r.error);
      el.appendChild(li);
    });
    const summary = document.getElementById('summary');
    if (summary) {
      summary.textContent = `${passed}/${results.length} passed`;
      summary.className = passed === results.length ? 'pass' : 'fail';
    }
  }

  FT.test = { test, assert, run, render, cases };
})(typeof window !== 'undefined' ? window : globalThis);
