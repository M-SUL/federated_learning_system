import './domshim.mjs';
import fs from 'fs';
const snap = JSON.parse(fs.readFileSync(new URL('./snap.json', import.meta.url),'utf8'));
const { VIEWS } = await import('./views.mjs');
let fails = 0;
const base = {snapshot:snap, liveStatus:'none', liveRun:null, liveInfo:null,
              multipleLive:[], raw:null, selected:{}};
for (const [k,v] of Object.entries(VIEWS)) {
  for (const [name, st] of [
    ['with data', {...base, view:k}],
    ['no snapshot', {...base, snapshot:null, view:k}],
  ]) {
    try { v.render(st); console.log('OK    ' + k.padEnd(12) + name); }
    catch(e){ fails++; console.log('FAIL  ' + k.padEnd(12) + name + ': ' + e.message); }
  }
}
process.exit(fails?1:0);
