import './domshim.mjs';
import fs from 'fs';
const { VIEWS } = await import('./views.mjs');
const snap = JSON.parse(fs.readFileSync(new URL('./snap.json', import.meta.url),'utf8'));
const opts = {strategy:['iid','dirichlet','dirichlet-legacy'],alpha:[0.1,0.3,0.5,1.0],
              norm:['batch','layer','none'],rounds:[2,3,5,10],seed:[42,43,44]};
const base = {snapshot:snap, liveStatus:'none', liveRun:null, liveInfo:null, multipleLive:[],
              raw:null, selected:{}, view:'console',
              launchForm:{strategy:'dirichlet',alpha:0.1,rounds:10,norm:'batch',seed:42},
              launchPending:false, launchError:null};
const cases = [
  ['launcher enabled',   {...base, launch:{enabled:true,available:true,options:opts,busy:null}}],
  ['launcher busy',      {...base, launch:{enabled:true,available:true,options:opts,
                                           busy:{run_id:'123',status:'running'}}}],
  ['launcher disabled',  {...base, launch:{enabled:false,disabled_reason:'not loopback',options:opts}}],
  ['flwr missing',       {...base, launch:{enabled:true,available:false,
                                           unavailable_reason:'flwr not found',options:opts}}],
  ['iid hides alpha',    {...base, launchForm:{...base.launchForm,strategy:'iid'},
                          launch:{enabled:true,available:true,options:opts,busy:null}}],
  ['launch error shown', {...base, launchError:'alpha: 0.7 is not allowed',
                          launch:{enabled:true,available:true,options:opts,busy:null}}],
  ['no launch info',     {...base, launch:null}],
];
let fails=0;
for (const [name, st] of cases) {
  try { VIEWS.console.render(st); console.log('OK    ' + name); }
  catch(e){ fails++; console.log('FAIL  ' + name + ': ' + e.message); }
}
process.exit(fails?1:0);
