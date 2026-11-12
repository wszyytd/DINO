/* Data-only viewer smoke test using a small DOM stub; no browser or model. */
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const file = process.argv[2];
if (!file) throw new Error('Usage: node tests/region_sequence_viewer_smoke.cjs output/preview/index.html');
const html = fs.readFileSync(file, 'utf8');
const script = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));
assert(!script.includes('__PAYLOAD__'), 'Payload must be embedded');
const noop = () => {};
class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.value = ''; this.textContent = ''; this.style = {};
  }
  append(...children) {
    this.children.push(...children);
    if (this.tagName === 'select' && !this.value && this.children.length) this.value = this.children[0].value;
  }
  replaceChildren(...children) { this.children = []; if (this.tagName === 'select') this.value = ''; this.append(...children); }
  getContext() { return new Proxy({}, {get: (_, key) => key === 'canvas' ? this : noop}); }
}
function launch(source) {
  const elements = {};
  let interval = null;
  for (const id of ['seed', 'method', 'split', 'route']) elements[id] = new Element('select');
  elements.method.value = 'memory';
  elements.step = new Element('input'); elements.step.value = 0;
  const context = vm.createContext({
    console, setInterval: fn => { interval = fn; return 1; }, clearInterval: () => { interval = null; },
    document: {
      getElementById: id => elements[id] ||= new Element(),
      createElement: tag => new Element(tag),
      createTextNode: text => ({textContent: text})
    },
    Image: class {
      constructor() { this.width = 1024; this.height = 768; }
      set src(value) { this._src = value; this.onload?.(); }
    }
  });
  vm.runInContext(source, context, {filename: file, timeout: 20000});
  return {context, elements, data: vm.runInContext('DATA', context), tick: () => interval?.()};
}
function exercise(page) {
  const {context, elements: e, data} = page;
  function check() {
    const r = data.routes.find(r => r.id === e.route.value), t = Number(e.step.value);
    assert(t >= 0 && t < r.records.length);
    assert.equal(e.reference.src, data.records[r.records[t]].reference_asset);
    assert.equal(JSON.parse(e.raw.textContent).record.id, r.records[t]);
    assert.equal(e.cars.children.length, r.identities.length);
    assert.equal(e.compare.children.length, 3);
    if (!data.preview && t === 1) assert(e.control.textContent.includes('synthetic_reset'));
    r.identities.forEach((uid, index) => {
      const expected = [], used = new Set();
      for (let s = 0; s <= t; s++) {
        const obs = data.records[r.records[s]].observations.find(o => o.identity === uid && o.write_allowed);
        if (obs && (!data.config.exact_dedup || !used.has(obs.key))) {
          expected.push([obs.crop, `第${s+1}步`]); used.add(obs.key);
        }
      }
      const shown = e.cars.children[index].children[2].children.map(a => [a.href, a.children[2].textContent]);
      assert.deepEqual(shown, expected, 'History must contain only matching, already-written crops');
    });
  }
  check();
  const firstRoute = e.route.value;
  const r = data.routes.find(r => r.id === firstRoute);
  for (let i = 0; i < r.records.length + 2; i++) { e.next.onclick(); check(); }
  assert.equal(e.route.value, firstRoute, 'Next must never switch route or reference');
  assert.equal(Number(e.step.value), r.records.length - 1);
  e.play.onclick(); page.tick(); assert.equal(e.play.textContent, '播放'); check();
  e.prev.onclick(); check();
  e.step.value = 0; e.step.oninput(); check();
  const alternatives = data.routes.filter(r => r.split === e.split.value);
  if (alternatives.length > 1) {
    e.route.value = alternatives[1].id; e.route.onchange(); check();
    assert.equal(Number(e.step.value), 0);
  }
  e.split.value = 'train'; e.split.onchange(); check();
  e.play.onclick(); assert.equal(e.play.textContent, '暂停');
  page.tick(); assert.equal(Number(e.step.value), 1); check();
  e.next.onclick(); assert.equal(e.play.textContent, '播放'); check();
  vm.runInContext('render()', context); check();
}

const preview = launch(script);
assert(preview.data.preview);
exercise(preview);
// Exercise the prediction branch with explicitly synthetic records. These are not experimental results.
const data = JSON.parse(JSON.stringify(preview.data));
data.preview = false; data.predictions = {'7': {}}; data.controls = {'7': {}};
for (const method of ['last', 'mean', 'memory']) {
  data.predictions['7'][method] = {};
  for (const r of data.routes) {
    const seen = new Set(), writes = {}, used = new Set();
    data.controls['7'][r.id] = {'1': [{control: 'synthetic_reset', choice_identity: null, control_correct: false}]};
    data.predictions['7'][method][r.id] = r.records.map((rid, step) => {
      const row = data.records[rid], current = row.observations.filter(o => o.write_allowed), written = [];
      for (const o of current) {
        if (!data.config.exact_dedup || !used.has(o.key)) {
          writes[o.identity] = (writes[o.identity] || 0) + 1; written.push(o.identity); used.add(o.key);
        }
        seen.add(o.identity);
      }
      const memory = Object.fromEntries(r.identities.map(uid => [uid, {
        seen: seen.has(uid), writes: writes[uid] || 0, slot_sha256: '0123456789abcdef', change_l2: step ? 0 : null
      }]));
      return {route_id: r.id, step, record_id: rid, identities: r.identities, memory,
        choice_identity: null, accepted_truth: seen.has(r.target_identity) ? r.target_identity : null,
        physical_truth: null, correct: !seen.has(r.target_identity),
        current_identities: current.map(o => o.identity), written_identities: written,
        probabilities: [...r.identities.map(() => 0), 1], scores: [...r.identities.map(() => -100), 0]};
    });
  }
}
const synthetic = script.replace(/^const DATA=.*;\r?\n/, 'const DATA=' + JSON.stringify(data).replaceAll('<', '\\u003c') + ';\n');
assert.notEqual(synthetic, script, 'Synthetic payload must replace preview payload');
const trained = launch(synthetic); exercise(trained);
assert(trained.elements.null.textContent.includes('100.00%'));
console.log('Viewer data smoke passed: route boundary, fixed reference, causal crop history, playback, controls and synthetic score display.');
console.log('This is a DOM-stub check, not a visual browser test or model evaluation.');
