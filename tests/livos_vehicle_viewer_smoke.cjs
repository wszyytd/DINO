/* DOM-stub checks of real saved preview plus synthetic result rendering. No model. */
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const file = process.argv[2];
if (!file) throw new Error('Usage: node tests/livos_vehicle_viewer_smoke.cjs output/preview/index.html');
const html = fs.readFileSync(file, 'utf8');
const source = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));
assert(!source.includes('__DATA__'));
function launch(script) {
  const elements = {};
  function item() {
    return {value: '0', textContent: '', children: [], hidden: false,
      appendChild(x) {this.children.push(x);}, replaceChildren() {this.children = [];}};
  }
  const element = id => elements[id] ||= item();
  let interval = null;
  const context = vm.createContext({console,
    document: {getElementById: element, createElement: item},
    setInterval: fn => {interval = fn; return 1;}, clearInterval: () => {interval = null;}});
  vm.runInContext(script, context, {filename: file, timeout: 20000});
  return {elements, data: vm.runInContext('DATA', context), tick: () => interval?.()};
}
function exercise(page) {
  const {elements: e, data} = page;
  function check() {
    const c = data.cases[Number(e.cases.value)], step = Number(e.slider.value);
    assert(step >= 0 && step < c.steps.length);
    assert.equal(e.rgb.src, c.steps[step].image);
    assert.equal(e.gt.src, c.steps[step].gt);
    assert.equal(e.history.children.length, step + 1);
    assert.deepEqual(e.history.children.map(i => i.src), Array.from(c.steps.slice(0, step + 1), i => i.image));
    assert.equal(e.pred.hidden, data.preview);
    if (!data.preview) assert(e.metric.textContent.includes('IoU'));
  }
  check();
  for (let i = 0; i < data.cases[0].steps.length + 2; i++) {e.next.onclick(); check();}
  e.play.onclick(); page.tick(); assert.equal(e.play.textContent, '播放'); check();
  e.prev.onclick(); check(); e.history.children[0].onclick(); check();
  assert.equal(Number(e.slider.value), 0);
  e.prev.onclick(); check(); assert.equal(Number(e.slider.value), 0);
  if (data.cases.length > 1) {e.cases.value = '1'; e.cases.onchange(); check();}
  e.play.onclick(); page.tick(); check(); assert.equal(Number(e.slider.value), 1);
  e.play.onclick(); assert.equal(e.play.textContent, '播放');
}
const preview = launch(source);
assert(preview.data.preview);
exercise(preview);
// Synthetic display values stay in this in-memory test, never saved as predictions.
const data = JSON.parse(JSON.stringify(preview.data));
data.preview = false;
data.summary = {overall: {scored_steps: 1, mean_visible_iou: .5, visible_success_rate: .5,
  absent_false_positive_rate: null, two_visible_wrong_rate: 0, recovery_rate: null}};
for (const c of data.cases) for (const step of c.steps) {
  step.prediction = step.image;
  step.metrics = {iou: step.target_pixels ? .5 : null, success: true,
    wrong_vehicle_overlap: false, absent_false_positive: false,
    state: {persistent_tensor_bytes: 1000}};
}
const marker = source.indexOf('\nconst el=');
assert(marker > 0);
const synthetic = 'const DATA=' + JSON.stringify(data).replaceAll('<', '\\u003c') + ';' + source.slice(marker);
exercise(launch(synthetic));
console.log('LiVOS vehicle viewer passed: case/step navigation, past-only history, playback and synthetic score display.');
console.log('DOM-stub check only; no browser visual test or model execution.');
