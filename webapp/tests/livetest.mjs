import './domshim.mjs';
import fs from 'fs';
const { newRaw, reduce, project } = await import('./store.mjs');
const { VIEWS } = await import('./views.mjs');
const snap = JSON.parse(fs.readFileSync(new URL('./snap.json', import.meta.url),'utf8'));
const lines = fs.readFileSync(new URL('./ev.jsonl', import.meta.url),'utf8').trim().split('\n');

let raw = newRaw('test');
for (const l of lines) reduce(raw, JSON.parse(l));
// replay the whole log again: must be a no-op (idempotent on seq)
const before = raw.events.length;
for (const l of lines) reduce(raw, JSON.parse(l));
console.log('events:', lines.length, '| idempotent on replay:', raw.events.length === before);

const run = project(raw, snap);
console.log('projected run:', run.slug, '| rounds:', run.curves.points.length,
            '| sites:', run.sites.length, '| macroF1:', run.headline.macro_f1);

// The architectural invariant: live and stored runs must be the same shape.
const stored = snap.runs.find(r => r.family === 'federated');
const missing = Object.keys(stored).filter(k => !(k in run));
console.log('keys missing from live run vs stored:', missing.length ? missing : 'none');

const st = {snapshot:snap, liveStatus:'running', liveRun:run, liveInfo:null,
            multipleLive:[], raw, selected:{}};
let fails=0;
for (const [k,v] of Object.entries(VIEWS)) {
  try { v.render({...st, view:k}); console.log('OK    ' + k + ' (live)'); }
  catch(e){ fails++; console.log('FAIL  ' + k + ' (live): ' + e.message); }
}
process.exit(fails || missing.length ? 1 : 0);
