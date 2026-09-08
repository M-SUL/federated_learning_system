/* Chart primitives. Every chart in the app is built from these.
 *
 * Hand-rolled SVG rather than a chart library: the console has to work during a
 * defence with no network, and the palette must match the matplotlib figures in
 * results/ exactly. Written as one primitives module so the five views cannot
 * each grow their own subtly different line-drawing code.
 */

export const NS = 'http://www.w3.org/2000/svg';

export function el(name, attrs = {}, children = []) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    n.setAttribute(k, String(v));
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

export function svg(width, height, attrs = {}) {
  return el('svg', {
    viewBox: `0 0 ${width} ${height}`, width: '100%', height,
    preserveAspectRatio: 'xMidYMid meet', ...attrs,
  });
}

/* ---- scales ---- */

export function linear(domain, range) {
  let [d0, d1] = domain;
  if (d0 === d1) { d0 -= 0.5; d1 += 0.5; }
  const [r0, r1] = range;
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.domain = [d0, d1]; f.range = range;
  return f;
}

export function log10(domain, range) {
  // Guards against a zero or negative lower bound: C grids start at 1e-4 but a
  // caller passing 0 would otherwise produce -Infinity and a blank chart.
  const d0 = Math.log10(Math.max(domain[0], Number.MIN_VALUE));
  const d1 = Math.log10(Math.max(domain[1], Number.MIN_VALUE * 10));
  const f = linear([d0, d1], range);
  const g = (v) => f(Math.log10(Math.max(v, Number.MIN_VALUE)));
  g.domain = domain; g.range = range;
  return g;
}

export function band(items, range, pad = 0.2) {
  const [r0, r1] = range;
  const step = (r1 - r0) / Math.max(items.length, 1);
  const w = step * (1 - pad);
  const f = (i) => r0 + i * step + (step - w) / 2;
  f.bandwidth = w; f.step = step;
  return f;
}

/* ---- ticks ---- */

export function niceTicks(d0, d1, count = 5) {
  if (d0 === d1) return [d0];
  const span = d1 - d0;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

export function logTicks(d0, d1) {
  const out = [];
  for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
    const v = Math.pow(10, e);
    if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
  }
  return out;
}

/* ---- axes ---- */

const TICK_LABEL = { 'font-size': 10, fill: 'var(--ink-muted)' };

export function axisLeft(scale, x, ticks, fmt = String) {
  const g = el('g');
  for (const t of ticks) {
    const y = scale(t);
    g.appendChild(el('text', { x: x - 8, y: y + 3.5, 'text-anchor': 'end', ...TICK_LABEL }, fmt(t)));
  }
  return g;
}

export function axisBottom(scale, y, ticks, fmt = String) {
  const g = el('g');
  for (const t of ticks) {
    g.appendChild(el('text', { x: scale(t), y: y + 16, 'text-anchor': 'middle', ...TICK_LABEL }, fmt(t)));
  }
  return g;
}

export function gridLines(scale, ticks, x0, x1) {
  const g = el('g');
  for (const t of ticks) {
    const y = scale(t);
    g.appendChild(el('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: 'var(--grid)', 'stroke-width': 1 }));
  }
  return g;
}

export function axisTitle(text, x, y, rotate = 0) {
  return el('text', {
    x, y, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink)',
    transform: rotate ? `rotate(${rotate} ${x} ${y})` : null,
  }, text);
}

/* ---- marks ---- */

export function linePath(points, xs, ys, xKey, yKey) {
  const d = points
    .filter((p) => p[yKey] !== null && p[yKey] !== undefined && Number.isFinite(p[yKey]))
    .map((p, i) => `${i ? 'L' : 'M'}${xs(p[xKey]).toFixed(2)},${ys(p[yKey]).toFixed(2)}`)
    .join(' ');
  return d || null;
}

export function line(points, xs, ys, xKey, yKey, color, width = 2) {
  const d = linePath(points, xs, ys, xKey, yKey);
  return d ? el('path', { d, fill: 'none', stroke: color, 'stroke-width': width,
                          'stroke-linejoin': 'round', 'stroke-linecap': 'round' }) : null;
}

export function dots(points, xs, ys, xKey, yKey, color, r = 3.5) {
  const g = el('g');
  for (const p of points) {
    const v = p[yKey];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    // A 2px surface ring keeps overlapping markers separable.
    g.appendChild(el('circle', { cx: xs(p[xKey]), cy: ys(v), r, fill: color,
                                 stroke: 'var(--surface)', 'stroke-width': 1.5 }));
  }
  return g;
}

export function bar(x, y, w, h, color, rx = 3) {
  // Round only the data end; the baseline end stays square so bars sit flat.
  const r = Math.min(rx, w / 2, Math.max(h, 0));
  return el('rect', { x, y, width: Math.max(w, 0), height: Math.max(h, 0),
                      rx: r, ry: r, fill: color });
}

export function valueLabel(text, x, y, anchor = 'middle') {
  return el('text', { x, y, 'text-anchor': anchor, 'font-size': 9.5, fill: 'var(--ink)' }, text);
}

/* ---- heatmap ---- */

/** Single-hue ramp. Sequential data is never a rainbow. */
export function blues(t) {
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(247 - c * 199);
  const g = Math.round(251 - c * 175);
  const b = Math.round(255 - c * 118);
  return `rgb(${r},${g},${b})`;
}

/**
 * Matrix heatmap with a value in every cell.
 * Cells carry raw counts, not normalized fractions: most off-diagonal cells here
 * are genuinely empty, and an integer 0 reads as "no samples" where "0.00" reads
 * like a formatting bug.
 */
export function heatmap({ rows, rowLabels, colLabels, cellSize = 44, normalizeRows = true,
                          labelWidth = 58, topLabels = true }) {
  const nR = rows.length, nC = rows[0] ? rows[0].length : 0;
  const top = topLabels ? 22 : 4;
  const w = labelWidth + nC * cellSize + 8;
  const h = top + nR * cellSize + 8;
  const s = svg(w, h, { class: 'heatmap' });

  const maxAll = Math.max(1, ...rows.flat());
  rows.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    row.forEach((v, j) => {
      const t = normalizeRows ? (total ? v / total : 0) : v / maxAll;
      const x = labelWidth + j * cellSize, y = top + i * cellSize;
      s.appendChild(el('rect', {
        x: x + 1, y: y + 1, width: cellSize - 2, height: cellSize - 2,
        fill: blues(t), rx: 2,
      }));
      s.appendChild(el('text', {
        x: x + cellSize / 2, y: y + cellSize / 2 + 4, 'text-anchor': 'middle',
        'font-size': 11, fill: t > 0.6 ? '#fff' : 'var(--ink)',
      }, String(v)));
    });
    s.appendChild(el('text', {
      x: labelWidth - 8, y: top + i * cellSize + cellSize / 2 + 4,
      'text-anchor': 'end', 'font-size': 10, fill: 'var(--ink)',
    }, rowLabels[i]));
  });

  if (topLabels) {
    colLabels.forEach((c, j) => s.appendChild(el('text', {
      x: labelWidth + j * cellSize + cellSize / 2, y: 14,
      'text-anchor': 'middle', 'font-size': 10, fill: 'var(--ink)',
    }, c)));
  }
  return s;
}

/** Horizontal stacked bar — one client's label composition. */
export function stackedBar(segments, width, height = 16) {
  const total = segments.reduce((a, s) => a + s.count, 0) || 1;
  const g = el('g');
  let x = 0;
  for (const seg of segments) {
    const w = (seg.count / total) * width;
    if (w > 0) {
      // 1px gap so adjacent segments stay separable without a border colour.
      g.appendChild(el('rect', { x, y: 0, width: Math.max(w - 1, 0.5), height,
                                 fill: seg.color, rx: 1 }));
    }
    x += w;
  }
  return g;
}

export function legend(items, { columns = 0 } = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'legend';
  if (columns) wrap.style.gridTemplateColumns = `repeat(${columns}, auto)`;
  for (const it of items) {
    const s = document.createElement('span');
    s.className = 'legend-item' + (it.dashed ? ' dashed' : '');
    s.innerHTML = `<i style="background:${it.color}"></i>${it.label}`;
    wrap.appendChild(s);
  }
  return wrap;
}
