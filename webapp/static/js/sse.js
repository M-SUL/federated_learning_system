/* EventSource wrapper.
 *
 * Three behaviours that are easy to get wrong and cost a hung tab each:
 *  - a finished run must be closed explicitly on `eof`, or EventSource
 *    reconnects to it forever;
 *  - a 204 means "no run at all" and EventSource will NOT retry, so we poll
 *    /api/runs and open a stream when one appears;
 *  - reconnects resume from Last-Event-ID automatically, and the reducer
 *    dedupes the overlap by seq.
 */

import { dispatch, getState, newRaw, project, reduce } from './store.js';

const EVENT_TYPES = [
  'run_start', 'round_start', 'client_train', 'agg_train', 'eval_fanout',
  'client_evaluate', 'agg_eval', 'global_eval', 'run_end', 'run_error',
];

let es = null;
let pollTimer = null;
let pending = false;
// The run we have already streamed. /api/runs reports the newest run as
// `current` even once it has finished (so history can be viewed), so without
// this we would reconnect to a finished run, replay it, hit eof, and poll
// again -- a refresh loop every few seconds that never settles.
let streamedRunId = null;

function flush() {
  // Batch repaints: a fast round can deliver 12 events in one tick, and we want
  // one render, not twelve.
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    const st = getState();
    dispatch({ liveRun: project(st.raw, st.snapshot) });
  });
}

function onEvent(ev) {
  let obj;
  try { obj = JSON.parse(ev.data); } catch { return; }
  const st = getState();
  if (!st.raw) dispatch({ raw: newRaw(obj.run_id || 'live') });
  reduce(getState().raw, obj);
  if (obj.type === 'run_end') dispatch({ liveStatus: 'finished' });
  else if (obj.type === 'run_error') dispatch({ liveStatus: 'failed' });
  flush();
}

function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

/**
 * Should the poller open a stream for what /api/runs just reported?
 *
 * Only for a run that is actually LIVE and has not already been streamed.
 * `current` names the newest run even once it has finished, so attaching on
 * `current` alone replays a finished run, hits eof, restarts the poll, and
 * repaints forever. Exported so that rule is testable without a browser.
 */
export function shouldAttach(runs, current, streamed) {
  const cur = (runs || []).find((x) => x.run_id === current);
  return !!(cur && cur.is_live && cur.run_id !== streamed);
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/runs');
      const d = await r.json();
      if (shouldAttach(d.runs, d.current, streamedRunId)) {
        stopPolling();
        connect();
      }
    } catch { /* server down; keep polling quietly */ }
  }, 3000);
}

export function connect() {
  if (es) { es.close(); es = null; }
  dispatch({ liveStatus: 'connecting' });

  // A 204 gives EventSource nothing to attach to, so probe first: this keeps
  // "no run yet" a quiet poll instead of a reconnect storm.
  fetch('/api/events', { method: 'HEAD' }).catch(() => {}).finally(() => {
    es = new EventSource('/api/events');

    es.addEventListener('hello', (ev) => {
      let info = null;
      try { info = JSON.parse(ev.data); } catch { /* ignore */ }
      const run = info && info.run;
      if (run) streamedRunId = run.run_id;
      dispatch({
        liveInfo: run,
        multipleLive: (info && info.multiple_live) || [],
        liveStatus: run ? (run.is_live ? 'running' : run.status) : 'none',
        raw: newRaw(run ? run.run_id : 'live'),
      });
    });

    for (const t of EVENT_TYPES) es.addEventListener(t, onEvent);

    es.addEventListener('eof', (ev) => {
      let reason = {};
      try { reason = JSON.parse(ev.data); } catch { /* ignore */ }
      // Without this close(), a finished run is reconnected to indefinitely.
      es.close(); es = null;
      const st = getState();
      dispatch({
        liveStatus: reason.reason === 'stale' ? 'crashed'
          : st.liveStatus === 'failed' ? 'failed' : 'finished',
      });
      startPolling();
    });

    es.onerror = () => {
      if (es && es.readyState === EventSource.CLOSED) {
        es = null;
        dispatch({ liveStatus: 'none' });
        startPolling();
      }
    };
  });
}

export function init() {
  fetch('/api/runs')
    .then((r) => r.json())
    .then((d) => {
      dispatch({ multipleLive: d.multiple_live || [] });
      if (d.current) connect(); else { dispatch({ liveStatus: 'none' }); startPolling(); }
    })
    .catch(() => dispatch({ liveStatus: 'none' }));
}
