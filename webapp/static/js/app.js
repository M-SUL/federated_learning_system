/* Boot, routing, and the header status pill. */

import { dispatch, getState, subscribe } from './store.js';
import { init as sseInit } from './sse.js';
import { VIEWS } from './views.js';

const STATUS_CLASS = {
  running: 'live', connecting: 'pending', finished: 'done',
  failed: 'bad', crashed: 'bad', none: 'idle',
};
const STATUS_LABEL = {
  running: 'Federation running', connecting: 'Connecting', finished: 'Run finished',
  failed: 'Run failed', crashed: 'Run ended unexpectedly', none: 'No run',
};

// Views own no state of their own; selections live in the store so a live event
// re-render never resets what the user picked.
window.__select = (patch) => dispatch({ selected: { ...getState().selected, ...patch } });

function renderTabs(state) {
  const nav = document.getElementById('tabs');
  nav.replaceChildren();
  for (const [key, v] of Object.entries(VIEWS)) {
    const b = document.createElement('button');
    b.className = 'tab' + (state.view === key ? ' on' : '');
    b.textContent = v.label;
    if (key === 'console' && state.liveStatus === 'running') {
      const dot = document.createElement('i');
      dot.className = 'dot';
      b.appendChild(dot);
    }
    b.addEventListener('click', () => { location.hash = key; });
    nav.appendChild(b);
  }
}

function renderStatus(state) {
  const pill = document.getElementById('status');
  pill.className = 'pill ' + (STATUS_CLASS[state.liveStatus] || 'idle');
  pill.textContent = STATUS_LABEL[state.liveStatus] || state.liveStatus;
}

function renderWarnings(state) {
  const box = document.getElementById('warnings');
  const w = state.snapshot ? state.snapshot.warnings : [];
  if (!w || !w.length) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren();
  const d = document.createElement('details');
  const s = document.createElement('summary');
  s.textContent = `Data provenance (${w.length})`;
  d.appendChild(s);
  const ul = document.createElement('ul');
  for (const line of w) {
    const li = document.createElement('li');
    li.textContent = line;
    ul.appendChild(li);
  }
  d.appendChild(ul);
  box.appendChild(d);
}

function render(state) {
  renderTabs(state);
  renderStatus(state);
  renderWarnings(state);
  const main = document.getElementById('main');
  const view = VIEWS[state.view] || VIEWS.sites;
  const scroll = main.scrollTop;
  main.replaceChildren(view.render(state));
  main.scrollTop = scroll;
}

function route() {
  const key = location.hash.replace('#', '');
  dispatch({ view: VIEWS[key] ? key : 'sites' });
}

async function boot() {
  subscribe(render);
  window.addEventListener('hashchange', route);
  route();
  try {
    const r = await fetch('/api/snapshot');
    dispatch({ snapshot: await r.json() });
  } catch (e) {
    document.getElementById('main').textContent =
      'Could not load results. Is the server still running?';
    return;
  }
  sseInit();
}

boot();
