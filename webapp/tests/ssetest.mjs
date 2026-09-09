import './domshim.mjs';
const { shouldAttach } = await import('./sse.mjs');

let fails = 0;
const t = (name, got, want) => {
  if (got === want) console.log('OK    ' + name);
  else { fails++; console.log(`FAIL  ${name}: got ${got}, want ${want}`); }
};

const live     = { run_id: 'A', is_live: true,  status: 'running' };
const finished = { run_id: 'A', is_live: false, status: 'finished' };
const crashed  = { run_id: 'A', is_live: false, status: 'crashed' };
const liveB    = { run_id: 'B', is_live: true,  status: 'running' };

// The regression: a finished run must NOT be reattached. Doing so replays it,
// ends in eof, restarts the poll, and repaints the console every few seconds.
t('finished run is not reattached',        shouldAttach([finished], 'A', 'A'), false);
t('finished run not attached even if new', shouldAttach([finished], 'A', null), false);
t('crashed run is not reattached',         shouldAttach([crashed],  'A', 'A'), false);
t('already-streamed live run is not redone', shouldAttach([live],   'A', 'A'), false);

t('a new live run IS attached',            shouldAttach([live],  'A', null), true);
t('a different live run IS attached',      shouldAttach([liveB, finished], 'B', 'A'), true);

t('no runs at all',                        shouldAttach([], null, null), false);
t('current names a run we do not have',    shouldAttach([live], 'ZZZ', null), false);
t('undefined runs list',                   shouldAttach(undefined, 'A', null), false);

process.exit(fails ? 1 : 0);
