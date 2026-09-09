import './domshim.mjs';
import fs from 'fs';
const { VIEWS } = await import('./views.mjs');

/** Text of a node built by h(): labels arrive as child text nodes. */
const textOf = (n) => (n.children || [])
  .map((c) => (c && c.text) ? c.text : '')
  .join('') + (n.textContent || '');
const snap = JSON.parse(fs.readFileSync(new URL('./snap.json', import.meta.url),'utf8'));
const opts = {strategy:['iid','dirichlet','dirichlet-legacy'],alpha:[0.1,0.3,0.5,1.0],
              norm:['batch','layer','none'],rounds:[2,3,5,10],seed:[42,43,44],
              clients:[2,3,5,8,10]};
const base = {snapshot:snap, liveStatus:'none', liveRun:null, liveInfo:null, multipleLive:[],
              raw:null, selected:{}, view:'console',
              launchForm:{strategy:'dirichlet',alpha:0.1,rounds:10,norm:'batch',seed:42,clients:5},
              launchPending:false, launchError:null};
const ready = {...base, launch:{enabled:true,available:true,options:opts,busy:null,
                                current_clients:5}};
const busy  = {...base, launch:{enabled:true,available:true,options:opts,
                                busy:{run_id:'123',status:'running'}}};
let fails=0;
const t=(name,fn)=>{ try{ const m=fn(); if(m){fails++;console.log('FAIL  '+name+': '+m);} 
  else console.log('OK    '+name);}catch(e){fails++;console.log('FAIL  '+name+': '+e.message);} };

for (const [name, st] of [
  ['renders when ready', ready], ['renders when busy', busy],
  ['launcher disabled',  {...base, launch:{enabled:false,disabled_reason:'not loopback',options:opts}}],
  ['flwr missing',       {...base, launch:{enabled:true,available:false,
                                           unavailable_reason:'flwr not found',options:opts}}],
  ['iid hides alpha',    {...ready, launchForm:{...base.launchForm,strategy:'iid'}}],
  ['error shown',        {...ready, launchError:'alpha: 0.7 is not allowed'}],
  ['no launch info',     {...base, launch:null}],
]) t(name, () => { VIEWS.console.render(st); return null; });

// The regression: controls must be INTERACTIVE when no run is in progress.
// A false boolean must omit the attribute entirely -- disabled="false" disables.
t('controls enabled when idle', () => {
  const root = VIEWS.console.render(ready);
  const selects = root.findAll((n) => n.tag === 'select');
  if (selects.length < 5) return `expected >=5 selects, got ${selects.length}`;
  const stuck = selects.filter((s) => s.isDisabled);
  if (stuck.length) return `${stuck.length} select(s) disabled while idle`;
  const btn = root.find((n) => n.tag === 'button' && /Start run/.test(textOf(n)));
  if (!btn) return 'no Start run button';
  if (btn.isDisabled) return 'Start run disabled while idle';
  return null;
});

t('controls disabled while a run is in progress', () => {
  const root = VIEWS.console.render(busy);
  const selects = root.findAll((n) => n.tag === 'select');
  if (!selects.length || !selects.every((s) => s.isDisabled)) return 'selects should be disabled';
  const btn = root.find((n) => n.tag === 'button' && /Start run/.test(textOf(n)));
  if (btn && !btn.isDisabled) return 'Start run should be disabled';
  return null;
});

t('site-count change is announced', () => {
  const changed = {...ready, launchForm:{...ready.launchForm, clients:10}};
  const root = VIEWS.console.render(changed);
  const warned = root.findAll((n) => n.tag === 'p' && /restarts the SuperLink/.test(textOf(n)));
  if (!warned.length) return 'no warning when the site count differs from the live config';
  const same = VIEWS.console.render(ready);
  const quiet = same.findAll((n) => n.tag === 'p' && /restarts the SuperLink/.test(textOf(n)));
  if (quiet.length) return 'warned even though the site count is unchanged';
  return null;
});

process.exit(fails?1:0);
