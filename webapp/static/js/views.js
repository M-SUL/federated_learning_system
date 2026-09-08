/* The five views. Each takes state and returns a DOM node.
 *
 * Views 1, 3 and 4 render fully with no run ever having been started -- that is
 * the answer to "dashboard opened before any training", and the fallback if
 * `flwr run` misbehaves during a demo.
 */

import {
  axisBottom, axisLeft, axisTitle, band, bar, dots, el, gridLines, heatmap,
  legend, line, linear, log10, logTicks, niceTicks, stackedBar, svg, valueLabel,
} from './svg.js';

const pct = (v) => (v === null || v === undefined ? '—' : (v * 100).toFixed(1) + '%');
const f4 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(4));
const gb = (b) => (b === null || b === undefined ? '—' : (b / 1e9).toFixed(3) + ' GB');
const mb = (b) => (b === null || b === undefined ? '—' : (b / 1e6).toFixed(1) + ' MB');

function h(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, String(v));
  }
  for (const c of [].concat(kids)) {
    if (c === null || c === undefined) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

function card(title, subtitle, body) {
  return h('section', { class: 'card' }, [
    h('header', {}, [h('h2', {}, title), subtitle ? h('p', { class: 'sub' }, subtitle) : null]),
    body,
  ]);
}

function empty(msg) { return h('p', { class: 'empty' }, msg); }

/* ================= 1. Sites ================= */

export function sitesView(state) {
  const snap = state.snapshot;
  if (!snap) return empty('Loading…');

  const slugs = Object.keys(snap.partitions);
  if (!slugs.length) return empty('No partition data found.');
  const slug = state.selected.partitionSlug && snap.partitions[state.selected.partitionSlug]
    ? state.selected.partitionSlug
    : slugs.find((s) => !s.startsWith('dirichlet-legacy')) || slugs[0];
  const p = snap.partitions[slug];

  const picker = h('div', { class: 'row wrap' }, slugs.map((s) => h('button', {
    class: 'chip' + (s === slug ? ' on' : ''),
    onclick: () => window.__select({ partitionSlug: s }),
  }, s.replace('_seed42', ''))));

  const maxShard = Math.max(...p.clients.map(
    (c) => c.label_counts.reduce((a, x) => a + x.count, 0)), 1);

  const rows = p.clients.map((c) => {
    const total = c.label_counts.reduce((a, x) => a + x.count, 0);
    const seen = c.label_counts.filter((x) => x.count > 0).length;
    const s = svg(320, 18);
    s.appendChild(stackedBar(c.label_counts, (total / maxShard) * 320, 16));
    return h('tr', {}, [
      h('td', {}, `Site ${c.site_id}`),
      h('td', { class: 'num' }, String(total)),
      h('td', { class: 'num' + (seen < 5 ? ' warn' : '') }, `${seen}/5`),
      h('td', { class: 'barcell' }, s),
      h('td', { class: 'tiny' }, c.label_counts.filter((x) => x.count > 0)
        .map((x) => `${x.class} ${x.count}`).join(' · ')),
    ]);
  });

  const table = h('table', { class: 'data' }, [
    h('thead', {}, h('tr', {}, ['Site', 'Samples', 'Classes', 'Composition', 'Detail']
      .map((t) => h('th', {}, t)))),
    h('tbody', {}, rows),
  ]);

  const counts = p.clients.map((c) => c.label_counts.map((x) => x.count));
  const hm = heatmap({
    rows: counts,
    rowLabels: p.clients.map((c) => `Site ${c.site_id}`),
    colLabels: snap.classes,
    normalizeRows: false,
  });

  const js = p.heterogeneity_js;
  const provenance = p.source === 'partitions.json'
    ? null
    : h('p', { class: 'note' },
      'Shard composition reconstructed from the local-only run — partitions.json '
      + 'has no entry for this configuration.');

  return h('div', {}, [
    card('Sites and data heterogeneity',
      'Each site is a simulated hospital holding data it never shares. '
      + 'Heterogeneity is the mean Jensen–Shannon divergence between a site’s label '
      + 'distribution and the global one; 0 means every site looks like the whole cohort.',
      h('div', {}, [
        picker,
        h('div', { class: 'statrow' }, [
          stat('Heterogeneity (JS)', js === null || js === undefined ? '—' : js.toFixed(3)),
          stat('Sites', String(p.clients.length)),
          stat('Sites missing ≥1 class',
            String(p.clients.filter((c) => c.label_counts.filter((x) => x.count > 0).length < 5).length)),
        ]),
        provenance,
        table,
        h('div', { class: 'chartwrap' }, hm),
        legend(snap.classes.map((c) => ({ label: c, color: snap.palette.classes[c] }))),
      ])),
  ]);
}

function stat(label, value, hint) {
  return h('div', { class: 'stat' }, [
    h('div', { class: 'stat-v' }, value),
    h('div', { class: 'stat-l' }, label),
    hint ? h('div', { class: 'stat-h' }, hint) : null,
  ]);
}

/* ================= 2. Training monitor ================= */

function curveChart(run, palette) {
  const c = run.curves;
  if (!c || !c.kind || !c.points.length) {
    return empty('This run has no training curve — logistic regression with a fixed '
      + 'C reports no per-step history.');
  }
  const W = 720, H = 300, M = { t: 16, r: 16, b: 42, l: 52 };
  const s = svg(W, H);
  const xKey = c.x.key;
  const xs0 = c.points.map((p) => p[xKey]);
  const xs = c.x.scale === 'log'
    ? log10([Math.min(...xs0), Math.max(...xs0)], [M.l, W - M.r])
    : linear([Math.min(...xs0), Math.max(...xs0)], [M.l, W - M.r]);
  const xticks = c.x.scale === 'log'
    ? logTicks(Math.min(...xs0), Math.max(...xs0))
    : niceTicks(Math.min(...xs0), Math.max(...xs0), 6).filter(Number.isInteger);

  // Metric series share a 0..1 axis; loss is on its own chart rather than a
  // second y-axis, which would be unreadable at these scales.
  const metricSeries = c.series.filter((sr) => sr.axis === 'metric');
  const ys = linear([0, 1], [H - M.b, M.t]);
  const yticks = niceTicks(0, 1, 5);
  s.appendChild(gridLines(ys, yticks, M.l, W - M.r));
  s.appendChild(axisLeft(ys, M.l, yticks, (v) => v.toFixed(1)));
  s.appendChild(axisBottom(xs, H - M.b, xticks,
    c.x.scale === 'log' ? (v) => v.toExponential(0) : String));
  s.appendChild(axisTitle(c.x.label, (M.l + W - M.r) / 2, H - 6));

  const colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7'];
  metricSeries.forEach((sr, i) => {
    if (sr.band) {
      const up = c.points.map((p) => ({ x: p[xKey], v: (p[sr.key] ?? 0) + (p[sr.band] ?? 0) }));
      const dn = [...c.points].reverse()
        .map((p) => ({ x: p[xKey], v: (p[sr.key] ?? 0) - (p[sr.band] ?? 0) }));
      const d = up.map((p, j) => `${j ? 'L' : 'M'}${xs(p.x)},${ys(p.v)}`).join(' ')
        + ' ' + dn.map((p) => `L${xs(p.x)},${ys(p.v)}`).join(' ') + ' Z';
      s.appendChild(el('path', { d, fill: colors[i % colors.length], opacity: 0.15 }));
    }
    const ln = line(c.points, xs, ys, xKey, sr.key, colors[i % colors.length]);
    if (ln) s.appendChild(ln);
    s.appendChild(dots(c.points, xs, ys, xKey, sr.key, colors[i % colors.length]));
  });

  // Round 0 is the untrained initial model, not a training step. Marking it
  // stops the first segment reading as a real improvement between two rounds.
  if (c.x.zero_is_init && xs0.includes(0)) {
    s.appendChild(el('line', { x1: xs(0), x2: xs(0), y1: M.t, y2: H - M.b,
                               stroke: 'var(--ink-muted)', 'stroke-dasharray': '3 3',
                               'stroke-width': 1 }));
    s.appendChild(el('text', { x: xs(0) + 5, y: M.t + 11, 'font-size': 9.5,
                               fill: 'var(--ink-muted)' }, 'init'));
  }

  const lossSeries = c.series.find((sr) => sr.axis === 'loss');
  const wrap = h('div', {}, [
    h('div', { class: 'chartwrap' }, s),
    legend(metricSeries.map((sr, i) => ({ label: sr.label, color: colors[i % colors.length] }))),
  ]);

  if (lossSeries && c.points.some((p) => p[lossSeries.key] !== undefined)) {
    const vals = c.points.map((p) => p[lossSeries.key]).filter((v) => Number.isFinite(v) && v > 0);
    if (vals.length) {
      const s2 = svg(W, 200);
      const M2 = { t: 14, r: 16, b: 38, l: 52 };
      // Log scale: the interesting fact is where the loss settles, and the
      // alpha=0.1 plateau is an order of magnitude above the IID one.
      const ys2 = log10([Math.min(...vals), Math.max(...vals)], [200 - M2.b, M2.t]);
      const t2 = logTicks(Math.min(...vals), Math.max(...vals));
      s2.appendChild(gridLines(ys2, t2, M2.l, W - M2.r));
      s2.appendChild(axisLeft(ys2, M2.l, t2, (v) => v.toExponential(0)));
      s2.appendChild(axisBottom(xs, 200 - M2.b, xticks, String));
      s2.appendChild(axisTitle('Test loss (log)', 14, 100, -90));
      const ln = line(c.points, xs, ys2, xKey, lossSeries.key, '#D55E00');
      if (ln) s2.appendChild(ln);
      s2.appendChild(dots(c.points, xs, ys2, xKey, lossSeries.key, '#D55E00'));
      wrap.appendChild(h('div', { class: 'chartwrap' }, s2));
    }
  }
  return wrap;
}

export function trainingView(state) {
  const snap = state.snapshot;
  if (!snap) return empty('Loading…');

  const runs = [...(state.liveRun ? [state.liveRun] : []), ...snap.runs];
  const id = state.selected.trainingRunId
    && runs.find((r) => r.id === state.selected.trainingRunId)
    ? state.selected.trainingRunId
    : (state.liveRun ? state.liveRun.id : (runs.find((r) => r.family === 'federated') || runs[0]).id);
  const run = runs.find((r) => r.id === id);

  const picker = h('div', { class: 'row wrap' }, runs.map((r) => h('button', {
    class: 'chip' + (r.id === id ? ' on' : '') + (r.id.startsWith('live/') ? ' live' : ''),
    onclick: () => window.__select({ trainingRunId: r.id }),
  }, r.label)));

  const ceiling = Math.max(...snap.runs.filter((r) => r.family === 'centralized')
    .map((r) => r.headline.macro_f1 || 0));

  const degraded = run.health && run.health.degraded
    ? h('p', { class: 'alert' },
      `Degraded run: rounds ${run.health.failed_rounds.join(', ')} had no successful `
      + 'client replies, so the global model did not update. These numbers are not comparable.')
    : null;

  const perSite = run.sites && run.sites.length
    ? h('div', {}, [
      h('h3', {}, 'Per-site replies'),
      h('div', { class: 'sitegrid' }, run.sites.map((s2) => h('div', { class: 'sitecard' }, [
        h('div', { class: 'sitehead' }, `Site ${s2.site_id}`),
        h('div', { class: 'tiny' }, `${s2.n_shard ?? '—'} samples · ${s2.n_classes_seen ?? '—'}/5 classes`),
        s2.global ? h('div', { class: 'tiny' }, `global macro-F1 ${f4(s2.global.macro_f1)}`) : null,
        s2.local_val ? h('div', { class: 'tiny warn' }, `local-val acc ${pct(s2.local_val.accuracy)}`) : null,
      ]))),
    ])
    : null;

  return h('div', {}, [
    card('Training monitor',
      'Global model quality per round, measured on the shared held-out test set. '
      + 'Macro-F1 saturates on this dataset, so the loss panel is where heterogeneity shows.',
      h('div', {}, [
        picker,
        degraded,
        h('div', { class: 'statrow' }, [
          stat('Final macro-F1', f4(run.headline.macro_f1)),
          stat('Final accuracy', pct(run.headline.accuracy)),
          stat('Centralized ceiling', f4(ceiling)),
          stat('Gap to ceiling', run.headline.macro_f1 === null ? '—'
            : f4(ceiling - run.headline.macro_f1)),
        ]),
        curveChart(run, snap.palette),
        perSite,
      ])),
  ]);
}

/* ================= 3. Federated vs isolated ================= */

export function comparisonView(state) {
  const snap = state.snapshot;
  if (!snap) return empty('Loading…');
  const rows = snap.comparisons.filter((c) => c.federated || c.local_only);
  const primary = rows.filter((c) => !c.is_legacy);
  const legacy = rows.filter((c) => c.is_legacy);

  const W = 760, H = 60 + primary.length * 74, M = { t: 20, r: 120, l: 190, b: 34 };
  const s = svg(W, H);
  const xs = linear([0, 1], [M.l, W - M.r]);
  const ticks = niceTicks(0, 1, 5);
  s.appendChild(gridLines(linear([0, primary.length], [M.t, H - M.b]), [], 0, 0));
  for (const t of ticks) {
    s.appendChild(el('line', { x1: xs(t), x2: xs(t), y1: M.t - 6, y2: H - M.b,
                               stroke: 'var(--grid)', 'stroke-width': 1 }));
    s.appendChild(el('text', { x: xs(t), y: H - M.b + 15, 'text-anchor': 'middle',
                               'font-size': 10, fill: 'var(--ink-muted)' }, t.toFixed(1)));
  }
  const ceiling = primary.length ? primary[0].centralized_ceiling : null;
  if (ceiling) {
    s.appendChild(el('line', { x1: xs(ceiling), x2: xs(ceiling), y1: M.t - 8, y2: H - M.b,
                               stroke: 'var(--ink-muted)', 'stroke-dasharray': '4 3' }));
    s.appendChild(el('text', { x: xs(ceiling), y: M.t - 12, 'text-anchor': 'end',
                               'font-size': 9.5, fill: 'var(--ink)' },
      `centralized ceiling ${ceiling.toFixed(3)}`));
  }

  primary.forEach((c, i) => {
    const y = M.t + i * 74;
    const label = c.strategy === 'iid' ? 'IID' : `Dirichlet α=${c.alpha}`;
    s.appendChild(el('text', { x: M.l - 12, y: y + 22, 'text-anchor': 'end',
                               'font-size': 11, fill: 'var(--ink)' }, label));
    s.appendChild(el('text', { x: M.l - 12, y: y + 36, 'text-anchor': 'end',
                               'font-size': 9, fill: 'var(--ink-muted)' },
      c.heterogeneity_js !== null && c.heterogeneity_js !== undefined
        ? `JS ${c.heterogeneity_js.toFixed(3)}` : ''));

    const pairs = [
      { v: c.federated && c.federated.macro_f1, color: snap.palette.methods.federated, name: 'Federated' },
      { v: c.local_only && c.local_only.macro_f1, color: snap.palette.methods.local_only, name: 'Isolated' },
    ];
    pairs.forEach((p, j) => {
      if (p.v === null || p.v === undefined) return;
      const by = y + 6 + j * 24;
      s.appendChild(bar(M.l, by, xs(p.v) - M.l, 18, p.color));
      s.appendChild(valueLabel(p.v.toFixed(3), xs(p.v) + 6, by + 13, 'start'));
    });
    if (c.gain !== null && c.gain !== undefined) {
      s.appendChild(el('text', { x: W - M.r + 52, y: y + 30, 'text-anchor': 'start',
                                 'font-size': 11, fill: 'var(--ink)',
                                 'font-weight': 600 }, `+${c.gain.toFixed(3)}`));
    }
  });

  const tbl = h('table', { class: 'data' }, [
    h('thead', {}, h('tr', {}, ['Partition', 'JS', 'Isolated', 'Federated', 'Gain', 'Source']
      .map((t) => h('th', {}, t)))),
    h('tbody', {}, rows.map((c) => h('tr', { class: c.is_legacy ? 'legacy' : '' }, [
      h('td', {}, (c.strategy === 'iid' ? 'IID' : `α=${c.alpha}`)
        + (c.is_legacy ? ' (legacy)' : '') + (c.norm !== 'batch' ? ` · ${c.norm}Norm` : '')),
      h('td', { class: 'num' }, c.heterogeneity_js == null ? '—' : c.heterogeneity_js.toFixed(3)),
      h('td', { class: 'num' }, c.local_only ? f4(c.local_only.macro_f1) : '—'),
      h('td', { class: 'num' }, c.federated ? f4(c.federated.macro_f1) : '—'),
      h('td', { class: 'num strong' }, c.gain == null ? '—' : '+' + c.gain.toFixed(3)),
      h('td', { class: 'tiny' }, c.partition_source || '—'),
    ]))),
  ]);

  return h('div', {}, [
    card('What aggregation recovers',
      'Federated and isolated models trained on identical shards. The gap is the '
      + 'cost of not collaborating — it widens as sites become less alike.',
      h('div', {}, [
        h('div', { class: 'chartwrap' }, s),
        legend([
          { label: 'Federated (FedAvg)', color: snap.palette.methods.federated },
          { label: 'Isolated (local only)', color: snap.palette.methods.local_only },
        ]),
        tbl,
        legacy.length ? h('p', { class: 'note' },
          'Legacy rows use the original hand-rolled Dirichlet partitioner — different '
          + 'shards, kept as a cross-implementation check, not a repeat of the same run.') : null,
      ])),
  ]);
}

/* ================= 4. Cost & privacy ledger ================= */

export function ledgerView(state) {
  const snap = state.snapshot;
  if (!snap) return empty('Loading…');
  const fed = state.liveRun
    || snap.runs.find((r) => r.family === 'federated' && r.config.norm === 'batch')
    || snap.runs.find((r) => r.family === 'federated');
  if (!fed) return empty('No federated run found.');

  const cost = fed.cost;
  const layer = snap.runs.find((r) => r.family === 'federated' && r.config.norm === 'layer');
  const batch = snap.runs.find((r) => r.family === 'federated'
    && r.config.norm === 'batch' && r.config.alpha === (layer && layer.config.alpha));

  const items = (cost.what_leaves_the_client || []).map((it) => h('tr',
    { class: it.protocol === false ? 'telemetry' : '' }, [
      h('td', {}, it.item),
      h('td', { class: 'tiny' }, it.kind),
      h('td', { class: 'num' }, it.count === null || it.count === undefined
        ? '—' : it.count.toLocaleString()),
      h('td', { class: 'tiny' }, it.protocol === false ? 'telemetry only' : 'protocol'),
      h('td', { class: 'tiny' }, it.note || ''),
    ]));

  const bnFinding = layer && batch ? h('div', { class: 'finding' }, [
    h('h3', {}, 'BatchNorm ships distributional statistics — and removing them helped'),
    h('p', {},
      'FedAvg averages BatchNorm’s running_mean and running_var across sites. Those are '
      + 'per-feature summaries of local expression values, which is exactly the kind of '
      + 'statistic federation exists to avoid sharing. Swapping to LayerNorm removes them '
      + 'from the payload entirely:'),
    h('div', { class: 'statrow' }, [
      stat('BatchNorm macro-F1', f4(batch.headline.macro_f1), `α=${batch.config.alpha}`),
      stat('LayerNorm macro-F1', f4(layer.headline.macro_f1), `α=${layer.config.alpha}`),
      stat('Difference',
        '+' + (layer.headline.macro_f1 - batch.headline.macro_f1).toFixed(4),
        'in LayerNorm’s favour'),
    ]),
    h('p', { class: 'tiny' },
      'The leaked statistic was not buying accuracy — the variant without it scored higher.'),
  ]) : null;

  return h('div', {}, [
    card('Cost and privacy ledger',
      'What federation costs in bandwidth, and precisely what leaves each site.',
      h('div', {}, [
        h('div', { class: 'statrow' }, [
          stat('Per model transfer', mb(cost.weights_bytes_per_transfer)),
          stat('Transfers', String(cost.transfers || 0), 'rounds × sites × 2'),
          stat('Total moved', gb(cost.weights_bytes_total)),
          stat('Raw data if pooled', mb(cost.raw_data_bytes_if_pooled), 'never transmitted'),
        ]),
        h('h3', {}, 'What leaves a site'),
        items.length ? h('table', { class: 'data' }, [
          h('thead', {}, h('tr', {}, ['Item', 'Kind', 'Count', 'Status', 'Note']
            .map((t) => h('th', {}, t)))),
          h('tbody', {}, items),
        ]) : empty('Connect a live run or select a stored federated run.'),
        bnFinding,
        h('p', { class: 'note' },
          'Patient records transmitted: zero. The expression matrix never leaves a site — '
          + 'that is the whole point of the exercise. The label-histogram row above is '
          + 'this dashboard’s own telemetry and is disclosed rather than hidden; '
          + 'emit-site-telemetry=false removes it.'),
      ])),
  ]);
}

/* ================= 5. Live console ================= */

const STATUS_TEXT = {
  none: 'No run detected', connecting: 'Connecting…', running: 'Running',
  finished: 'Finished', failed: 'Failed', crashed: 'Ended unexpectedly',
};

export function consoleView(state) {
  const run = state.liveRun;
  const status = state.liveStatus;

  if (!run) {
    return card('Live federation console', 'Streams a running federation, round by round.',
      h('div', {}, [
        h('p', { class: 'empty' }, 'No run to show yet.'),
        h('p', {}, 'Start one in another terminal:'),
        h('pre', {}, 'cd fl_rnaseq\nflwr run . local-simulation'),
        h('p', { class: 'tiny' },
          'The dashboard never starts training itself — it only reads the event log, '
          + 'so nothing here can affect a run. It will pick up a new run within a few seconds.'),
      ]));
  }

  const live = run._live || {};
  const rounds = live.rounds || [];
  const cur = rounds.length ? rounds[rounds.length - 1].round : 0;

  const chips = h('div', { class: 'sitegrid' }, (run.sites || []).map((s) => h('div',
    { class: 'sitecard ' + (s.ok ? 'ok' : 'err') }, [
      h('div', { class: 'sitehead' }, `Site ${s.site_id}`),
      h('div', { class: 'tiny' }, `${s.n_shard ?? '—'} samples · ${s.n_classes_seen ?? '—'}/5 classes`),
      h('div', { class: 'tiny' }, s.ok ? 'replying normally' : 'errors reported'),
    ])));

  const log = h('div', { class: 'eventlog' }, (state.raw ? state.raw.events : [])
    .slice(-120).reverse().map((e) => h('div', { class: 'ev ev-' + e.type }, [
      h('span', { class: 'ev-r' }, e.round === undefined || e.round === null ? '·' : `r${e.round}`),
      h('span', { class: 'ev-t' }, e.type),
      h('span', { class: 'ev-d' }, describeEvent(e)),
    ])));

  return h('div', {}, [
    card('Live federation console',
      run.slug + ' — ' + (STATUS_TEXT[status] || status),
      h('div', {}, [
        state.multipleLive.length ? h('p', { class: 'alert' },
          `${state.multipleLive.length + 1} runs are live at once. Showing the newest; `
          + 'streams are never merged.') : null,
        status === 'crashed' ? h('p', { class: 'alert' },
          `The run ended unexpectedly at round ${cur} — no completion event was written. `
          + 'Everything received so far is kept below.') : null,
        h('div', { class: 'statrow' }, [
          stat('Round', `${cur}${live.numRounds ? ' / ' + live.numRounds : ''}`),
          stat('Sites', String((run.sites || []).length)),
          stat('Macro-F1', f4(run.headline.macro_f1)),
          stat('Client errors', String(run.health.n_client_errors)),
        ]),
        chips,
        curveChart(run, state.snapshot ? state.snapshot.palette : null),
        h('h3', {}, 'Event stream'),
        log,
      ])),
  ]);
}

function describeEvent(e) {
  switch (e.type) {
    case 'run_start': return `${e.slug} · ${e.num_clients} sites · ${e.num_rounds} rounds`;
    case 'global_eval': return `acc ${f4(e.accuracy)} · macro-F1 ${f4(e.macro_f1)} · loss ${f4(e.loss)}`;
    case 'client_train':
    case 'client_evaluate':
      return e.ok
        ? `site ${e.site_id ?? '?'} · ${Object.entries(e.metrics || {})
            .map(([k, v]) => `${k} ${typeof v === 'number' ? v.toFixed(4) : v}`).join(' · ')}`
        : `site ${e.site_id ?? '?'} ERROR ${(e.error && e.error.reason || '').slice(0, 90)}`;
    case 'agg_train':
    case 'agg_eval': return `${e.n_ok} ok · ${e.n_error} failed`;
    case 'round_start': return `${(e.node_ids || []).length} sites dispatched`;
    case 'eval_fanout': return `${(e.node_ids || []).length} sites evaluating`;
    case 'run_end': return 'completed';
    case 'run_error': return `${e.exc_type}: ${(e.message || '').slice(0, 90)}`;
    default: return '';
  }
}

export const VIEWS = {
  sites: { label: 'Sites', render: sitesView },
  training: { label: 'Training', render: trainingView },
  comparison: { label: 'Federated vs isolated', render: comparisonView },
  ledger: { label: 'Cost & privacy', render: ledgerView },
  console: { label: 'Live console', render: consoleView },
};
