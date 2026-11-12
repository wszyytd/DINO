/* Exercise the saved viewer with a DOM stub; no browser or model execution. */
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const file = process.argv[2];
if (!file) throw new Error('Usage: node tests/fixed_height_viewer_smoke.cjs output/preview/index.html');
const html = fs.readFileSync(file, 'utf8');
const source = html.slice(html.indexOf('<script>')+8, html.lastIndexOf('</script>'));
assert(!source.includes('__DATA__'));
function launch(script) {
  const elements = {};
  const element = id => elements[id] ||= {value:'', textContent:'', innerHTML:'', disabled:false};
  element('case').value='0'; element('seed').value='7'; element('method').value='memory';
  let interval = null;
  const context = vm.createContext({console,
    document:{getElementById:element},
    setInterval: fn => {interval=fn; return 1;}, clearInterval: () => {interval=null;}
  });
  vm.runInContext(script, context, {filename:file, timeout:20000});
  return {elements, data:vm.runInContext('DATA',context), tick:()=>interval?.()};
}
function exercise(page) {
  const {elements:e,data} = page;
  function check() {
    const c=data.cases[Number(e.case.value)], t=Number(e.step.value);
    assert(t>=0 && t<c.records.length);
    assert.equal(e.ref.src,c.reference_asset);
    assert.equal(e.whole.src,data.records[c.records[t]].image_asset);
    const shown=[...e.history.innerHTML.matchAll(/data-step="(\d+)"/g)].map(m=>Number(m[1]));
    assert.equal(shown.length,2*(t+1));
    assert(shown.every(step=>step<=t),'History must not include future crops');
    if (!data.preview) assert(e.cars.innerHTML.includes('fixture_hash'));
  }
  check(); const c=data.cases[0];
  for(let i=0;i<c.records.length+2;i++){e.next.onclick();check();}
  assert.equal(e.case.value,'0'); assert.equal(Number(e.step.value),c.records.length-1);
  e.play.onclick(); page.tick(); assert.equal(e.play.textContent,'播放');check();
  e.prev.onclick();check();
  const before=Number(e.step.value); e.seed.value='17';e.seed.onchange();check();
  e.method.value='mean';e.method.onchange();check();assert.equal(Number(e.step.value),before);
  if(data.cases.length>1){e.case.value='1';e.case.onchange();check();assert.equal(Number(e.step.value),0);}
  e.step.value=0;e.step.oninput();e.prev.onclick();check();assert.equal(Number(e.step.value),0);
  e.play.onclick();page.tick();check();assert.equal(Number(e.step.value),1);
  e.next.onclick();assert.equal(e.play.textContent,'播放');check();
}
const preview=launch(source); assert(preview.data.preview);exercise(preview);
// Explicitly synthetic values exercise result rendering only, never written as model results.
const data=JSON.parse(JSON.stringify(preview.data));data.preview=false;data.results={};
for(const method of ['last','mean','memory']) for(const seed of [7,17,27]) {
  data.results[`${method}_${seed}`]=data.cases.flatMap(c=>c.records.map((id,step)=>({
    case_id:c.id,step,identities:c.identities,choice_identity:c.target_identity,correct:true,
    scores:[1,0,-1],probabilities:[.6,.3,.1],single_frame:{choice_identity:null},
    state:c.identities.map(identity=>({identity,count:step+1,sha256:'fixture_hash',change_l2:step?1:null}))
  })));
}
const marker=source.indexOf(';const $=');assert(marker>0);
const synthetic='const DATA='+JSON.stringify(data).replaceAll('<','\\u003c')+source.slice(marker);
exercise(launch(synthetic));
console.log('Fixed-height viewer checks passed: fixed reference, sequence boundaries, past-only history, playback, and synthetic score display.');
console.log('DOM-stub check only; no model or visual browser test.');
