/* Single state object, plus the reducer that folds live SSE events into it.
 *
 * The one architectural rule: `state.liveRun` has the SAME shape as any entry in
 * `snapshot.runs`. `project()` below is what guarantees it. That means each view
 * has exactly one render path and never needs to know whether its data came from
 * a stored metrics.json or from an event that arrived 40ms ago -- otherwise the
 * live path becomes a second, divergent, far less tested renderer.
 */

const state = {
  snapshot: null,
  liveStatus: 'none',   // none | connecting | running | finished | failed | crashed
  liveRun: null,        // a normalized run, shape-identical to snapshot.runs[i]
  liveInfo: null,       // run metadata from the `hello` frame
  multipleLive: [],
  raw: null,
  view: 'sites',
  selected: { partitionSlug: null, comparisonKey: null, trainingRunId: null },
  // Launcher: `launch` is the server's whitelist and current availability;
  // `launchForm` is the chosen values, which the server re-validates regardless.
  launch: null,
  launchForm: { strategy: 'dirichlet', alpha: 0.1, rounds: 10, norm: 'batch', seed: 42 },
  launchPending: false,
  launchError: null,
};

const subscribers = new Set();

export function getState() { return state; }
export function subscribe(fn) { subscribers.add(fn); return () => subscribers.delete(fn); }

export function dispatch(patch) {
  Object.assign(state, patch);
  for (const fn of subscribers) fn(state);
}

export function newRaw(runId) {
  return { runId, lastSeq: -1, start: null, rounds: new Map(), sites: new Map(), events: [] };
}

/* ---- reducer ---- */

function mkRound(r) {
  return { round: r, accuracy: null, macro_f1: null, loss: null,
           per_class: null, confusion: null, train: [], evaluate: [], t: null };
}

export function reduce(raw, ev) {
  // Idempotent on seq: a reconnect replays from Last-Event-ID and may overlap,
  // and a manual refresh replays the whole file.
  if (!ev || typeof ev.seq !== 'number' || ev.seq <= raw.lastSeq) return raw;
  raw.lastSeq = ev.seq;
  raw.events.push(ev);
  if (raw.events.length > 400) raw.events.splice(0, raw.events.length - 400);

  if (ev.type === 'run_start') { raw.start = ev; return raw; }

  if (ev.round !== undefined && ev.round !== null) {
    // Lazily created on first reference, NOT on round_start: the first event of
    // every run is global_eval for round 0, which has no round_start before it.
    if (!raw.rounds.has(ev.round)) raw.rounds.set(ev.round, mkRound(ev.round));
  }
  const r = ev.round !== undefined && ev.round !== null ? raw.rounds.get(ev.round) : null;

  switch (ev.type) {
    case 'global_eval':
      Object.assign(r, { accuracy: ev.accuracy, macro_f1: ev.macro_f1, loss: ev.loss,
                         per_class: ev.per_class, confusion: ev.confusion_matrix, t: ev.t });
      break;
    case 'client_train':
    case 'client_evaluate': {
      const phase = ev.type === 'client_train' ? 'train' : 'evaluate';
      r[phase].push(ev);
      const id = ev.site_id !== null && ev.site_id !== undefined ? ev.site_id : ev.node_id;
      const site = raw.sites.get(id) || { site_id: id, node_id: ev.node_id, history: [],
                                          label_counts: null, n_train: null, n_classes_seen: null,
                                          errors: 0, state: 'idle' };
      site.node_id = ev.node_id;
      if (!ev.ok) site.errors += 1;
      if (ev.site) {
        site.n_train = ev.site.n_train ?? site.n_train;
        site.n_classes_seen = ev.site.n_classes_seen ?? site.n_classes_seen;
        if (Array.isArray(ev.site.label_counts) && Array.isArray(ev.site.label_classes)) {
          site.label_counts = ev.site.label_classes.map((c, i) => ({
            class: c, count: ev.site.label_counts[i],
          }));
        }
      }
      site.state = ev.ok ? (phase === 'train' ? 'trained' : 'evaluated') : 'error';
      site.history.push({ round: ev.round, phase, ok: ev.ok, metrics: ev.metrics || {},
                          seconds: (ev.site || {}).seconds ?? null, error: ev.error });
      raw.sites.set(id, site);
      break;
    }
    default:
      break;
  }
  return raw;
}

/* ---- projection into the shared run shape ---- */

const CLASS_COLORS = {
  BRCA: '#0072B2', COAD: '#E69F00', KIRC: '#009E73', LUAD: '#D55E00', PRAD: '#CC79A7',
};

export function project(raw, snapshot) {
  if (!raw || !raw.start) return null;
  const s = raw.start;
  const rounds = [...raw.rounds.values()].sort((a, b) => a.round - b.round);
  const withEval = rounds.filter((r) => r.macro_f1 !== null);
  const latest = withEval[withEval.length - 1] || null;
  const classes = s.classes || Object.keys(CLASS_COLORS);

  const perClass = latest && latest.per_class
    ? classes
        .map((c) => ({ class: c, color: CLASS_COLORS[c], ...(latest.per_class[c] || {}) }))
        .filter((e) => (e.support || 0) > 0)
    : null;

  const nErrors = [...raw.sites.values()].reduce((a, s2) => a + s2.errors, 0);

  return {
    id: `live/${s.slug}`,
    family: 'federated',
    slug: s.slug,
    experiment: 'federated_fedavg',
    label: `LIVE — ${s.slug}`,
    timestamp: null,
    git_commit: null,
    env: {},
    config: { ...(s.config || {}), raw: s.config || {} },
    headline: latest
      ? { accuracy: latest.accuracy, macro_f1: latest.macro_f1, weighted_f1: null,
          balanced_accuracy: null, macro_auroc: null, loss: latest.loss }
      : { accuracy: null, macro_f1: null, weighted_f1: null, balanced_accuracy: null,
          macro_auroc: null, loss: null },
    per_class: perClass,
    per_class_present: !!perClass,
    classes_missing_from_predictions: [],
    confusion: { present: !!(latest && latest.confusion), labels: classes,
                 rows: latest ? latest.confusion : null },
    curves: {
      kind: 'rounds',
      x: { key: 'round', label: 'FedAvg round', scale: 'linear', zero_is_init: true },
      series: [
        { key: 'macro_f1', label: 'Macro-F1', axis: 'metric', band: null },
        { key: 'accuracy', label: 'Accuracy', axis: 'metric', band: null },
        { key: 'loss', label: 'Test loss', axis: 'loss', band: null },
      ],
      points: withEval.map((r) => ({ round: r.round, accuracy: r.accuracy,
                                     macro_f1: r.macro_f1, loss: r.loss })),
    },
    sites: [...raw.sites.values()]
      .sort((a, b) => a.site_id - b.site_id)
      .map((st) => ({
        site_id: st.site_id, ok: st.errors === 0, status: st.errors ? 'error' : 'ok',
        error: null, n_train: st.n_train, n_val: null,
        n_shard: st.label_counts ? st.label_counts.reduce((a, c) => a + c.count, 0) : null,
        n_classes_seen: st.n_classes_seen,
        label_counts: (st.label_counts || []).map((c) => ({ ...c, color: CLASS_COLORS[c.class] })),
        train_loss: null, seconds: null, global: null, local_val: null,
      })),
    site_summary: null,
    model: {
      kind: 'MLP',
      params: (s.model || {}).params,
      bytes_per_model_fp32: (s.model || {}).bytes_per_model_fp32,
      mb_per_model: (s.model || {}).mb_per_model,
      total_gb: (s.model || {}).total_gb,
      raw: s.model || {},
    },
    health: { degraded: false, failed_rounds: [], n_client_errors: nErrors },
    cost: {
      weights_bytes_per_transfer: (s.model || {}).bytes_per_model_fp32,
      // Counts only transfers that have actually happened, so the ledger ticks
      // up live rather than showing the projected total from round 0.
      transfers: rounds.reduce((a, r) => a + r.train.length + r.evaluate.length, 0),
      weights_bytes_total: (s.model || {}).bytes_per_model_fp32
        * rounds.reduce((a, r) => a + r.train.length + r.evaluate.length, 0),
      total_gb: null,
      raw_data_bytes_if_pooled: 640 * 20264 * 4,
      raw_data_leaves_client: false,
      what_leaves_the_client: [],
    },
    partition_slug: s.slug,
    artifacts: [],
    _live: { rounds, numRounds: s.num_rounds, numClients: s.num_clients },
  };
}
