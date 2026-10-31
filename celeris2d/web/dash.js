// the project dashboard (project.html): project, rail, job pages, forms
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmt = v => typeof v === 'number' ? (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : String(+v.toFixed(3))) : esc(v);
const file = p => `/api/project/file?path=${encodeURIComponent(p)}`;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const THEME = () => ({surface: css('--surface'), well: css('--well'), line: css('--line'), axis: css('--axis'), grid: css('--line'),
                      ink: css('--ink'), ink2: css('--ink2'), ink3: css('--ink3'), font: css('--font')});
// the page's Plotly look: a titled card plot in the figures' dark theme
function plotLayout(T, title, extra = {}) {
  const ax = {gridcolor: T.line, zeroline: false, linecolor: T.axis, tickfont: {size: 10.5}, title: {font: {size: 11}}};
  const L = {paper_bgcolor: T.surface, plot_bgcolor: T.surface, font: {family: T.font, size: 11, color: T.ink2}, height: 270,
    margin: {l: 52, r: 14, t: 34, b: 42}, legend: {orientation: 'h', x: 1, xanchor: 'right', y: 1.13, font: {size: 11}},
    hoverlabel: {bgcolor: T.well, bordercolor: T.line, font: {color: T.ink, family: T.font}},
    title: {text: title, x: 0.01, xanchor: 'left', y: 0.97, font: {size: 12.5, color: T.ink}}};
  Object.assign(L, extra);
  ['xaxis', 'yaxis'].forEach(a => { L[a] = Object.assign({}, ax, extra[a] || {}, {title: Object.assign({}, ax.title, (extra[a] || {}).title || {})}); });
  return L;
}
const MAPJOBS = ['Refine3D', 'LocalRefine', 'InitialModel', 'Sharpen', 'Subtract', 'ModeMaps', 'MaskCreate', 'Polish', 'CtfRefine'];
const CLASSJOBS = ['Class2D', 'HeteroRefine', 'Class3D', 'Select'];
const MOVIEJOBS = ['MotionCtf', 'Pick', 'Extract'];
const LIVE3D = ['InitialModel', 'Refine3D', 'HeteroRefine', 'Class3D', 'LocalRefine'];
// per-micrograph statistics of a MotionCtf -> Pick -> Extract chain, shown together: [key, label, unit, digits]
const MICSTATS = [['defocus_a', 'Defocus', 'Å', 0], ['astigmatism_a', 'Astigmatism', 'Å', 0], ['ctf_fit_a', 'CTF fit resolution', 'Å', 2],
  ['ctf_score', 'CTF fit score', '', 3], ['drift_a', 'Total drift', 'Å', 1], ['local_rms_a', 'Local motion (rms)', 'Å', 2], ['tilt_deg', 'Ice tilt', '°', 1],
  ['picks', 'Picks', '', 0], ['candidates', 'Pick candidates', '', 0], ['particles', 'Particles', '', 0]];
let jobs = [], queue = {running: [], queued: [], waiting: []}, autoSelect = true, current = null, tab = 'plots', types = {};
let pollTimer = null, logTimer = null, liveTimer = null, viewer = null;

async function api(url, body) {
  const r = await fetch(url, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}

// ------------------------------------------------------------------ project, queue, list
async function load() {
  let d;
  try { d = await api('/api/project'); } catch (e) { $('#pname').textContent = 'No project open'; $('#ppath').textContent = e.message; return; }
  jobs = d.jobs;
  $('#pname').textContent = d.root.split(/[\\/]/).filter(Boolean).pop();
  $('#ppath').textContent = d.root;
  try { const q = await api('/api/project/queue'); queue = q; autoSelect = q.auto_select; } catch (e) {}
  renderPulse(); renderAsks(); renderQueue(); renderList();
  clearTimeout(pollTimer);
  const busy = queue.running.length + queue.queued.length;
  pollTimer = setTimeout(async () => { const before = JSON.stringify(jobs.map(j => j.status)); await load();
    if (current && JSON.stringify(jobs.map(j => j.status)) !== before) showJob(current, false); }, busy ? 4000 : 20000);
}
function renderPulse() {
  const r = queue.running.length, q = queue.queued.length, w = queue.waiting.length;
  $('#pulse').innerHTML = `<span class="d ${r ? 'on' : ''}"></span>` +
    (r ? `running ${esc(queue.running[0].slice(0, 3))}` : 'idle') + (q ? `, ${q} queued` : '') + (w ? `, <b style="color:var(--you)">${w} waiting for you</b>` : '');
}
function renderAsks() {
  const w = queue.waiting.map(n => jobs.find(j => j.name === n)).filter(Boolean);
  $('#asks').innerHTML = w.length ? `<h2>Waiting for you</h2>` + w.map(j => `
    <button class="ask" data-n="${esc(j.name)}"><strong>${j.type === 'ManualPick' ? 'Pick particles by hand' : 'Choose the classes to keep'}</strong>
      <span>${esc(j.name)} · from ${esc((j.inputs || [])[0] || '')}</span></button>`).join('') : '';
  $('#asks').style.display = w.length ? '' : 'none';
  document.querySelectorAll('.ask').forEach(b => b.onclick = () => showJob(b.dataset.n));
}
function renderQueue() {
  const rows = [...queue.running.map(n => [n, 'running']), ...queue.queued.map(n => [n, 'queued'])];
  $('#queue').innerHTML = `<h2>Queue <span class="muted" style="font-weight:400">${rows.length ? rows.length + ' job' + (rows.length > 1 ? 's' : '') : 'empty'}</span></h2>` +
    rows.map(([n, s]) => `<div class="q"><span class="who" data-n="${esc(n)}">${icon((jobs.find(x => x.name === n) || {}).type)}<span class="dot ${s} ${s === 'running' ? 'pulse' : ''}"></span>${esc(n)}</span>` +
      (s === 'queued' ? `<button class="quiet" data-c="${esc(n)}" title="take it out of the queue">Cancel</button>` : `<button class="quiet" data-s="${esc(n)}" title="end the running job">Stop</button>`) + '</div>').join('');
  $('#queue').style.display = rows.length ? '' : 'none';
  document.querySelectorAll('#queue .who').forEach(e => e.onclick = () => showJob(e.dataset.n));
  document.querySelectorAll('#queue [data-c]').forEach(b => b.onclick = async () => {
    try { await api('/api/project/cancel', {job: b.dataset.c}); } catch (e) { alert(e.message); }
    load();
  });
  document.querySelectorAll('#queue [data-s]').forEach(b => b.onclick = async () => {
    if (!confirm(`Stop ${b.dataset.s}? Its process ends now; Restart runs it again.`)) return;
    try { await api('/api/project/stop', {job: b.dataset.s}); } catch (e) { alert(e.message); }
    load();
  });
}
// job types: a category (colour) and a glyph (shape) each, so a job reads at a glance in every view of the rail
const KIND = {Import: 'mic', MotionCtf: 'mic', Curate: 'mic', Pick: 'ptc', ManualPick: 'ptc', Extract: 'ptc', ImportParticles: 'ptc',
  Class2D: 'd2', Select: 'd2', InitialModel: 'd3', Refine3D: 'd3', HeteroRefine: 'd3', Class3D: 'd3', LocalRefine: 'd3', SymExpand: 'd3',
  Subtract: 'd3', CtfRefine: 'd3', Polish: 'd3', Modes3D: 'd3', ModeMaps: 'd3', Sharpen: 'map', MaskCreate: 'map', MapScore: 'map'};
const GLYPH = {        // 16 x 16, stroked
  Import: 'M8 2v8M4.5 6.5 8 10l3.5-3.5M2.5 11v2.5h11V11',
  MotionCtf: 'M8 8m-1.6 0a1.6 1.6 0 1 0 3.2 0a1.6 1.6 0 1 0-3.2 0M8 8m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0M8 8m-6.3 0a6.3 6.3 0 1 0 12.6 0a6.3 6.3 0 1 0-12.6 0',
  Curate: 'M2 3h12l-4.5 5.5V13l-3 1.2V8.5z',
  Pick: 'M8 8m-4.5 0a4.5 4.5 0 1 0 9 0a4.5 4.5 0 1 0-9 0M8 1v3M8 12v3M1 8h3M12 8h3',
  ManualPick: 'M3 2l9 5-4 1.2L6.4 12z M8.2 8.4l3.6 4.4',
  Extract: 'M2 5V2h3M11 2h3v3M14 11v3h-3M5 14H2v-3M6 6h4v4H6z',
  ImportParticles: 'M8 1.5v6M5.5 5 8 7.5 10.5 5M4 11.5a1 1 0 1 0 .01 0M8 12.5a1 1 0 1 0 .01 0M12 11.5a1 1 0 1 0 .01 0',
  Class2D: 'M2 2h5v5H2zM9 2h5v5H9zM2 9h5v5H2zM9 9h5v5H9z',
  Select: 'M2 2h12v12H2zM5 8.2l2.2 2.2L11.3 6',
  InitialModel: 'M8 1.5 14 5v6l-6 3.5L2 11V5zM2 5l6 3.5L14 5M8 8.5v6',
  Refine3D: 'M8 1.5 14 5v6l-6 3.5L2 11V5zM2 5l6 3.5L14 5M8 8.5v6M5 6.8v4.5M11 6.8v4.5',
  HeteroRefine: 'M5.5 2 10 4.5v5L5.5 12 1 9.5v-5zM10.5 5.5 15 8v5l-4.5 2.5L6 13',
  Class3D: 'M4 1.5 7 3v3.5L4 8 1 6.5V3zM12 1.5 15 3v3.5L12 8 9 6.5V3zM8 8.5l3 1.5v3.5L8 15l-3-1.5V10z',
  LocalRefine: 'M8 1.5 14 5v6l-6 3.5L2 11V5zM8 8m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
  SymExpand: 'M8 2v12M2.8 5l10.4 6M13.2 5 2.8 11',
  Subtract: 'M8 1.5 14 5v6l-6 3.5L2 11V5zM5.5 8.5h5',
  CtfRefine: 'M1 8c1.5-5 3-5 4.5 0s3 5 4.5 0 3-5 4.5 0',
  Polish: 'M8 1.5 9.6 6.4 14.5 8 9.6 9.6 8 14.5 6.4 9.6 1.5 8 6.4 6.4z',
  Modes3D: 'M1.5 8h13M4 5 1.5 8 4 11M12 5l2.5 3-2.5 3M8 3v10',
  ModeMaps: 'M2 3h3.5v10H2zM6.3 3h3.5v10H6.3zM10.5 3H14v10h-3.5z',
  Sharpen: 'M8 1.5 14.5 14h-13zM8 6.5v3.5',
  MaskCreate: 'M8 2c4 0 6 2.6 6 5.6S11.5 14 8 14 2 11.6 2 8.2 4 2 8 2z',
  MapScore: 'M2.5 14V9M6.2 14V5M9.8 14V7M13.5 14V2',
};
const icon = t => `<span class="ico k-${KIND[t] || 'x'}" aria-hidden="true"><svg viewBox="0 0 16 16"><path d="${GLYPH[t] || 'M3 3h10v10H3z'}"/></svg></span>`;
// the project as its processing line: stages in order, newest job first within a stage
const STAGES = [
  ['Micrographs', ['Import', 'MotionCtf', 'Curate', 'Pick', 'ManualPick', 'Extract', 'ImportParticles']],
  ['2D classes', ['Class2D', 'Select']],
  ['3D', ['InitialModel', 'Refine3D', 'HeteroRefine', 'Class3D', 'LocalRefine', 'SymExpand', 'Subtract', 'CtfRefine', 'Polish', 'Modes3D', 'ModeMaps']],
  ['Maps', ['Sharpen', 'MaskCreate', 'MapScore']],
];
function stageOf(t) { const i = STAGES.findIndex(([, ts]) => ts.includes(t)); return i < 0 ? STAGES.length : i; }
function headline(i, js) {
  const fin = js.filter(j => j.status === 'finished');
  const num = (k, re) => { const m = String(k || '').match(re); return m ? m[1] : null; };
  if (i === 0) {
    const mc = fin.find(j => j.type === 'MotionCtf'), ex = fin.find(j => j.type === 'Extract');
    return [mc && num(mc.key, /([\d,]+) mics/) + ' micrographs', ex && num(ex.key, /([\d,]+) ptcls/) + ' particles'].filter(Boolean).join(', ');
  }
  if (i === 1) { const s = fin.find(j => j.type === 'Select'); return s ? num(s.key, /([\d,]+) kept/) + ' particles kept' : ''; }
  const r = fin.map(j => parseFloat(num(j.key, /([\d.]+) Å/))).filter(x => x > 0);
  return r.length ? `best ${Math.min(...r).toFixed(2)} Å` : '';
}
let railView = 'list';
try { railView = localStorage.getItem('celeris-rail') || 'list'; } catch (e) {}
function renderList() {
  const q = $('#q').value.trim().toLowerCase();
  const rows = jobs.filter(j => !q || `${j.name} ${j.type} ${(types[j.type] || {}).name || ''} ${j.status} ${j.label || ''} ${j.final ? 'final' : ''}`.toLowerCase().includes(q))
    .slice().sort((a, b) => b.id - a.id);                     // newest first in every view
  document.querySelectorAll('#rv button').forEach(b => b.classList.toggle('on', b.dataset.v === railView));
  $('.wrap').classList.toggle('flowing', railView === 'flow');
  const row = j => `<div class="job ${esc(j.status)} ${j.name === current ? 'sel' : ''}" data-n="${esc(j.name)}" tabindex="0" title="${esc((types[j.type] || {}).name || j.type)} · ${esc(j.status)}">
      ${icon(j.type)}<div class="t">${esc(j.type)}<span class="n">${String(j.id).padStart(3, '0')}</span>${j.final ? '<span class="fin" title="' + esc(j.final_note || 'final') + '">★</span>' : ''}</div>
      <div class="k" title="${esc(j.key || j.status)}"><span class="dot ${esc(j.status)} ${j.status === 'running' ? 'pulse' : ''}"></span>${esc(j.status === 'finished' ? (j.key || 'finished') : j.status)}</div>
      <div class="l">${esc(j.label || '')}</div></div>`;
  let html;
  if (railView === 'stages') {
    const groups = [...STAGES.map(([n]) => [n, []]), ['Other', []]];
    rows.forEach(j => groups[stageOf(j.type)][1].push(j));
    html = groups.filter(([, js]) => js.length).map(([n, js]) => `
      <section class="stg"><h3>${esc(n)}<span class="hl">${esc(headline(STAGES.findIndex(([x]) => x === n), js))}</span></h3>
        ${js.map(row).join('')}</section>`).join('');
  } else if (railView === 'flow' && !q) html = flowRail(rows, row);
  else html = `<div class="plain">${rows.map(row).join('')}</div>`;
  $('#list').innerHTML = html || `<div class="empty">${q ? 'No job matches.' : 'No jobs yet: start with New job, Import movies.'}</div>`;
  document.querySelectorAll('.job').forEach(e => { e.onclick = () => showJob(e.dataset.n); e.onkeydown = ev => { if (ev.key === 'Enter') showJob(e.dataset.n); }; });
}
// the jobs as a flowchart: one row per job (newest on top), each job a node on its own lane, a line from every input
// up to the jobs made from it (a straight lane where a job feeds the next one)
function flowRail(rows, row) {
  const at = new Map(rows.map((j, r) => [j.name, r]));
  const kids = new Map(rows.map(j => [j.name, []]));
  rows.forEach(j => (j.inputs || []).forEach(p => { if (kids.has(p)) kids.get(p).push(j.name); }));
  const span = j => [Math.min(at.get(j.name), ...kids.get(j.name).map(k => at.get(k))), at.get(j.name)];
  const lane = new Map(), ends = [];                          // ends[l] = [last row used, its job]
  [...rows].sort((a, b) => span(a)[0] - span(b)[0] || at.get(a.name) - at.get(b.name)).forEach(j => {
    const [s, e] = span(j);
    let l = ends.findIndex(([end, who]) => end < s || (end === s && kids.get(j.name).includes(who)));   // touching: my newest child, straight on
    if (l < 0) { l = ends.length; ends.push(null); }
    ends[l] = [e, j.name]; lane.set(j.name, l);
  });
  const H = 46, W = 13, x0 = 14, nl = Math.max(1, ends.length), gut = x0 + nl * W + 4;
  const X = l => x0 + l * W, Y = r => r * H + H / 2;
  let paths = '';
  rows.forEach(j => (j.inputs || []).forEach(p => {
    if (!at.has(p)) return;
    const xp = X(lane.get(p)), yp = Y(at.get(p)), xc = X(lane.get(j.name)), yc = Y(at.get(j.name));
    paths += xp === xc ? `<path d="M${xp} ${yp}V${yc}"/>` : `<path d="M${xp} ${yp}V${yc + 12}Q${xp} ${yc} ${xc + (xp > xc ? 6 : -6)} ${yc}H${xc}"/>`;
  }));
  const nodes = rows.map(j => `<circle class="k-${KIND[j.type] || 'x'} ${esc(j.status)}" cx="${X(lane.get(j.name))}" cy="${Y(at.get(j.name))}" r="4.5"/>`).join('');
  return `<div class="flowr" style="--gut:${gut}px"><svg class="flowsvg" width="${gut}" height="${rows.length * H}" aria-hidden="true"><g class="edges">${paths}</g>${nodes}</svg>
    ${rows.map(row).join('')}</div>`;
}
$('#q').oninput = renderList;
document.querySelectorAll('#rv button').forEach(b => b.onclick = () => { railView = b.dataset.v; try { localStorage.setItem('celeris-rail', railView); } catch (e) {} renderList(); });

// ------------------------------------------------------------------ a job
function numbers(j) {
  const s = j.summary || {}, sh = typeof s.sharpened === 'object' && s.sharpened ? s.sharpened : {};
  const out = [];
  const add = (v, unit, cap) => { if (v !== undefined && v !== null && v !== '' && typeof v !== 'object') out.push([v, unit, cap]); };
  const res = s.resolution_solvent_mask_a ?? (j.type === 'Sharpen' ? s.resolution_a : undefined) ?? s.resolution_a;
  add(res, 'Å', 'resolution, FSC 0.143');
  add(s.resolution_unmasked_a, 'Å', 'unmasked');
  add(sh.resolution_a, 'Å', 'sharpened');
  add(s.b_factor ?? sh.b_factor, 'Å²', 'B-factor');
  add(s.particles_used ?? s.particles ?? s.picks, '', s.particles_used ? 'particles used' : s.particles ? 'particles' : 'picks');
  add(s.micrographs ?? s.movies, '', s.micrographs ? 'micrographs' : 'movies');
  add(s.classes && typeof s.classes !== 'object' ? s.classes : undefined, '', 'classes');
  add(s.seconds ? (s.seconds >= 120 ? Math.round(s.seconds / 60) : Math.round(s.seconds)) : undefined, s.seconds >= 120 ? 'min' : 's', 'run time');
  return out.slice(0, 7);
}
async function showJob(name, scroll = true) {
  current = name; renderList();
  clearTimeout(logTimer);
  let j;
  try { j = await api('/api/project/job?job=' + encodeURIComponent(name)); } catch (e) { $('#main').innerHTML = `<div class="empty err">${esc(e.message)}</div>`; return; }
  if (current !== j.name) return;
  history.replaceState(null, '', '#' + j.name.slice(0, 3));
  const meta = jobs.find(x => x.name === j.name) || {};
  const s = j.summary || {};
  const warns = s.warnings || [];
  const canRerun = !['Import', 'Select'].includes(j.type);
  const runLabel = ['failed', 'cancelled'].includes(j.status) ? 'Restart' : ['MotionCtf', 'Pick', 'Extract'].includes(j.type) ? 'Continue with new movies' : 'Run again';
  const tname = (types[j.type] || {}).name || j.type, dur = s.seconds ? (s.seconds >= 120 ? Math.round(s.seconds / 60) + ' min' : Math.round(s.seconds) + ' s') : '';
  const stateText = {running: 'running', finished: dur ? 'finished in ' + dur : 'finished', waiting: 'waiting for you', queued: 'queued', failed: 'failed', cancelled: 'stopped'}[j.status] || j.status;
  const from = (j.lineage || []).length > 1 ? (jobs.find(x => x.name === j.name) || {}).inputs || [] : [];
  const fromText = from.map(n => { const x = jobs.find(y => y.name === n); return x ? `${(types[x.type] || {}).name || x.type} ${n.slice(0, 3)}` : n; }).join(' and ');
  const kp = ['Import', 'MotionCtf', 'Pick', 'Extract', 'ManualPick'].includes(j.type) ? [] : numbers(j);
  // the usual next step: the first job type (in processing order) that takes this job as its main input
  const NEXT = {Import: 'MotionCtf', MotionCtf: 'Pick', Pick: 'Extract', ManualPick: 'Extract', Extract: 'Class2D', Select: 'InitialModel',
    InitialModel: 'Refine3D', Refine3D: 'Sharpen', HeteroRefine: 'Refine3D', Class3D: 'Refine3D', CtfRefine: 'Refine3D', Polish: 'Refine3D', LocalRefine: 'Sharpen'};
  const nt = j.status === 'finished' && types[NEXT[j.type]] ? types[NEXT[j.type]] : null;
  const nin = nt ? (nt.inputs || []).find(i => i.types.includes(j.type)) : null;
  $('#main').innerHTML = `
    <div class="jhead">
      ${icon(j.type).replace('class="ico', 'class="ico big')}
      <div class="title"><h1>${esc(tname)} <span class="jn">${esc(j.name.slice(0, 3))}</span>${j.final ? ' <span class="fin" title="' + esc(j.final_note || '') + '">★ final</span>' : ''}</h1>
        <div class="sub"><span class="state ${esc(j.status)}"><span class="dot ${esc(j.status)} ${j.status === 'running' ? 'pulse' : ''}"></span>${esc(stateText)}</span>
          ${meta.label ? `<span>${esc(meta.label)}</span>` : ''}${fromText ? `<span class="muted">from ${esc(fromText)}</span>` : ''}</div></div>
      <div class="acts">
        ${j.status === 'running' ? `<button id="stop" class="stopb" title="end the job's process now; it is kept, marked stopped, and can be restarted">Stop</button>` : ''}
        ${['queued', 'waiting'].includes(j.status) ? `<button id="cancel">Cancel</button>` : ''}
        ${canRerun ? `<button id="rerun" class="${['failed', 'cancelled'].includes(j.status) ? 'go' : ''}" ${['running', 'queued', 'waiting'].includes(j.status) ? 'disabled' : ''} title="a new job with the same inputs and settings, through the queue">${runLabel}</button>` : ''}
        ${types[j.type] ? `<button id="edit" title="the New job form with this job's inputs and settings">Change settings…</button>` : ''}
        ${nt && nin ? `<button id="nextj" class="go" title="${esc(nt.blurb || '')}">Next: ${esc(nt.name || nt.type)}</button>` : ''}
        <details class="menu"><summary aria-label="more actions">⋯</summary><div class="menu-list">
          <button id="final" title="final jobs and everything they came from make the publication report">${j.final ? '★ Unmark final' : '☆ Mark final (for the report)'}</button>
          <button id="del" ${j.status === 'running' ? 'disabled' : ''} title="move the job to the project's Trash folder">Delete…</button></div></details>
      </div>
    </div>
    ${kp.length ? `<div class="kpis">${kp.map(([v, u, c]) => `<div class="kpi"><div class="v">${u === 'Å' && typeof v === 'number' ? v.toFixed(2) : fmt(v)}${u ? `<small>${u}</small>` : ''}</div><div class="c">${esc(c)}</div></div>`).join('')}</div>` : ''}
    ${j.error || s.error ? `<div class="warn err">${esc(j.error || s.error)}</div>` : ''}
    ${warns.map(w => `<div class="warn">${esc(w)}</div>`).join('')}
    <div id="stage"></div>
    <div class="tabs" role="tablist">${['plots', 'log', 'settings', 'files', 'lineage'].map(t => `<button data-t="${t}" class="${t === tab ? 'on' : ''}" role="tab">${{plots: 'Results', files: 'Files', log: 'Log', settings: 'Settings', lineage: 'Made from'}[t]}</button>`).join('')}</div>
    <div class="pane" id="pane"></div>`;
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => { tab = b.dataset.t; document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b)); renderPane(j); });
  renderStage(j);
  renderPane(j);
  const on = (id, fn) => { const b = $('#' + id); if (b) b.onclick = fn; };
  document.querySelectorAll('.jhead .menu-list button').forEach(b => b.addEventListener('click', () => b.closest('details').open = false));
  on('rerun', async () => { try { const d = await api('/api/project/rerun', {job: j.name}); current = d.job; await load(); showJob(d.job); } catch (e) { alert(e.message); } });
  on('edit', () => jobForm(j.type, j.params));
  on('nextj', () => jobForm(nt.type, {[nin.name]: j.id}));
  on('stop', async () => {
    if (!confirm(`Stop ${j.name}? Its process ends now; what it wrote so far stays, and Restart runs it again.`)) return;
    try { await api('/api/project/stop', {job: j.name}); await load(); showJob(j.name, false); } catch (e) { alert(e.message); }
  });
  on('cancel', async () => { try { await api('/api/project/cancel', {job: j.name}); await load(); showJob(j.name, false); } catch (e) { alert(e.message); } });
  on('final', async () => {
    let note = '';
    if (!j.final) { note = prompt('A note for the report (for example "final map, Figure 2"):', ''); if (note === null) return; }
    await api('/api/project/final', {job: j.name, final: !j.final, note}); await load(); showJob(j.name, false);
  });
  on('del', async () => {
    if (!confirm(`Delete ${j.name}? It moves to the project's Trash folder.`)) return;
    try { await api('/api/project/delete', {job: j.name}); }
    catch (e) {
      if (!/input of/.test(e.message)) { alert(e.message); return; }
      if (!confirm(e.message + '\n\nDelete it anyway? Those jobs keep their own results.')) return;
      try { await api('/api/project/delete', {job: j.name, force: true}); } catch (e2) { alert(e2.message); return; }
    }
    current = null; $('#main').innerHTML = '<div class="empty"><h1>Deleted</h1><p>It is in the project\'s Trash folder.</p></div>'; load();
  });
  if (scroll) $('#main').scrollTop = 0;
}

// the data stage: what the job made, first
function renderStage(j) {
  const st = $('#stage');
  if (viewer) { viewer.dispose(); viewer = null; }
  if (j.type === 'Import') return stageImport(st, j);
  if (j.type === 'ManualPick' && window.CelerisPicker) return CelerisPicker.mount(st, {job: j, api, file, css, theme: THEME()});
  if (j.type === 'Extract' && j.status !== 'queued') return stageExtract(st, j);
  if (MOVIEJOBS.includes(j.type) && j.status !== 'queued') return stageMicPanel(st, j);
  if (j.status === 'running') return stageLive(st, j);
  if ((CLASSJOBS.includes(j.type) || (j.summary || {}).class_table) && !['running', 'queued', 'cancelled', 'failed'].includes(j.status)) return stageGallery(st, j);
  const maps = j.files.filter(f => /\.mrc$/.test(f.path) && !/half[12]|(^|_)mask\.mrc$|localres|nu_cutoff|_frames/.test(f.path.split('/').pop()));
  if (MAPJOBS.includes(j.type) && maps.length) return stageVolume(st, j, maps);
  if (j.movie_files && j.movie_files.length) return stageMics(st, j);
  st.innerHTML = '';
}
// a running job: time, the last lines of its log and its newest figure, as they come
function stageLive(st, j) {
  st.className = 'stage';
  const t0 = (j.params || {}).run_started, skew = Date.now() / 1000 - (j.now || Date.now() / 1000);
  st.innerHTML = `<div class="live"><div class="runstrip" id="rstrip"><div class="rsteps" id="rsteps"><div class="rstep"><div class="rsh"><b>Starting</b></div><div class="bar"><div style="width:2%;background:var(--you)"></div></div></div></div>
        <div class="reta"><b id="reta">–</b><span id="lel" class="muted"></span></div></div>
      <div class="lhead"><span class="muted">Last line:</span> <span id="llast" class="lastline"></span></div>
      ${LIVE3D.includes(j.type) ? `<div id="l3d"><div class="muted" style="padding:14px 4px">The map appears here once the job has made one.</div></div>` : ''}
      ${j.type === 'Class2D' ? `<div id="lgal"><div class="muted" style="padding:18px 4px">The classes appear here after the first iteration.</div></div>` : ''}
      <div class="lbody nofig" id="lbody"><pre class="log ltail" id="ltail">waiting for the first lines of the log…</pre>
      <figure class="lfig" id="lfig"></figure></div></div>`;
  let offset = 0, text = '', lastFig = '', n = 0, liveAt = 0, at3d = 0;
  // the run strip: the job's stages, done / current (with its step of n) / to come, and the time left at the current pace
  const STAGES3D = {InitialModel: [['classes', 'Class averages'], ['sgd', 'Particle SGD'], ['refine', 'Refinement'], ['class3d', '3D classes']],
    Refine3D: [['refine', 'Refinement']], LocalRefine: [['refine', 'Local refinement']], HeteroRefine: [['refine', 'Heterogeneous refinement'], ['class3d', 'Classes']],
    Class3D: [['class3d', '3D classification']]};
  const strip = (stages, cur, it, nIt, hist) => {
    if (!$('#rsteps')) return;
    const ci = Math.max(0, stages.findIndex(([k]) => k === cur));
    $('#rsteps').innerHTML = stages.map(([k, lab], i) => {
      const done = i < ci, now = i === ci, pct = done ? 100 : now ? Math.max(2, 100 * it / Math.max(nIt, 1)) : 0;
      return `<div class="rstep ${done ? 'done' : now ? 'now' : ''}"><div class="rsh"><span class="rmark">${done ? '✓' : i + 1}</span><b>${esc(lab)}</b><span class="muted">${now ? `${it} of ${nIt}` : done ? 'done' : ''}</span></div>
        <div class="bar"><div style="width:${pct.toFixed(1)}%;background:${done ? 'var(--good)' : 'var(--you)'}"></div></div></div>`;
    }).join('');
    const h = (hist || []).filter(x => x.stage === cur && x.t);
    const el = t0 ? Math.max(0, Date.now() / 1000 - skew - t0) : 0;
    if ((h.length > 2 || (!hist && el && it > 1)) && nIt > it) {
      const dt = h.length > 2 ? (h[h.length - 1].t - h[0].t) / Math.max(h[h.length - 1].it - h[0].it, 1) : el / it, left = (nIt - it) * dt;
      $('#reta').textContent = (left >= 90 ? '~' + Math.round(left / 60) + ' min' : '~' + Math.round(left) + ' s') + ' left' + (ci < stages.length - 1 ? ' in this stage' : '');
    }
  };
  const live3d = async () => {                       // a 3D job's current maps (live.py), as they change: every model
    const d = await api('/api/project/live3d?job=' + encodeURIComponent(name));
    if (!d.maps || d.updated === at3d || current !== name || !$('#l3d')) return;
    at3d = d.updated;
    strip(STAGES3D[j.type] || [[d.stage, d.stage]], d.stage, d.it, d.n_iter, d.history);
    const stage = {sgd: 'Particle SGD', refine: 'Refinement', classes: 'From the class averages', class3d: '3D classification'}[d.stage] || d.stage;
    const K = d.maps.length, cnt = d.counts || [], tot = cnt.reduce((a, b) => a + (b || 0), 0);
    const views = await Promise.all(d.maps.map(m => api('/api/project/mapviews?path=' + encodeURIComponent(m)).catch(() => null)));
    if (current !== name || !$('#l3d')) return;
    const tile = (x, cls = '') => `<figure class="${cls}"><img src="${x.src}" alt="${esc(x.label)}"><figcaption>${esc(x.label.replace('Fourier section', 'Fourier'))}</figcaption></figure>`;
    $('#l3d').innerHTML = `<div class="gal-tools"><div class="sum">${esc(stage)}, step <b>${d.it}</b> of ${d.n_iter} · ${K > 1 ? K + ' classes · ' : ''}shown at ${d.apix.toFixed(2)} Å/px</div>
        <button class="quiet" id="l3dv">Show the current map${K > 1 ? 's' : ''} in 3D</button></div>
      <div class="l3m ${K > 1 ? 'multi' : ''}">${views.map((v, k) => v ? `<div class="l3card"><div class="l3t"><b>${K > 1 ? 'Class ' + (k + 1) : 'Map'}</b>
          ${cnt[k] != null ? `<span>${fmt(cnt[k])} particles${tot ? ' · ' + (100 * cnt[k] / tot).toFixed(1) + '%' : ''}</span>` : ''}
          ${(d.res || [])[k] != null ? `<span>${d.res[k].toFixed(1)} Å</span>` : ''}</div>
          <div class="l3row">${(K > 1 ? v.views.slice(0, 3) : v.views).map(x => tile(x)).join('')}${(v.fourier || []).slice(0, K > 1 ? 1 : 3).map(x => tile(x, 'four')).join('')}</div></div>` : '').join('')}</div>
      <div class="l3charts"><div id="l3h"></div>${K > 1 ? '<div id="l3c"></div>' : ''}</div>`;
    $('#l3dv').onclick = () => openViewer($('#stage'), j, d.maps.map(path => ({path})), null);
    const P = await CelerisPlots.plotly(), T = THEME(), h = d.history || [];
    if (!$('#l3h') || !h.length) return;
    const nm = Math.max(...h.map(x => (x.res || []).length), 1), cols = [css('--data'), css('--you'), css('--good'), css('--serious'), css('--k-d3'), css('--k-map'), css('--k-mic'), T.ink2];
    const step = h.map((_, i) => i + 1), lab = h.map(x => x.stage + ' ' + x.it);
    const tr = Array.from({length: nm}, (_, m) => ({type: 'scatter', mode: 'lines+markers', name: nm > 1 ? 'class ' + (m + 1) : 'resolution',
      x: step, y: h.map(x => (x.res || [])[m]), text: lab, line: {color: cols[m % cols.length], width: 1.8}, marker: {size: 4}, hovertemplate: '%{text}: %{y:.2f} Å<extra></extra>'}));
    const SN = {classes: 'class averages', sgd: 'SGD', refine: 'refinement', class3d: '3D classes'};
    const cuts = h.map((x, i) => i && x.stage !== h[i - 1].stage ? i + 0.5 : null).filter(x => x != null);
    const starts = [0.5, ...cuts];
    P.react($('#l3h'), tr, plotLayout(T, '<b>Resolution</b> per step (lower is better)', {showlegend: nm > 1,
      shapes: cuts.map(x => ({type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: {color: T.axis, width: 1, dash: 'dot'}})),
      annotations: starts.map((x, i) => ({x, xanchor: 'left', y: 1, yref: 'paper', yanchor: 'top', showarrow: false, text: SN[h[Math.ceil(x) - 1].stage] || h[Math.ceil(x) - 1].stage, font: {size: 10.5, color: T.ink3}})),
      xaxis: {title: {text: 'step'}}, yaxis: {title: {text: 'Å'}}}), {displaylogo: false, responsive: true});
    if (K > 1 && $('#l3c')) {
      const hc = h.filter(x => x.counts && x.counts.length === K);
      const tc = Array.from({length: K}, (_, m) => ({type: 'scatter', mode: 'lines', stackgroup: 'one', groupnorm: 'percent', name: 'class ' + (m + 1),
        x: hc.map(x => h.indexOf(x) + 1), y: hc.map(x => x.counts[m]), text: hc.map(x => x.stage + ' ' + x.it), line: {color: cols[m % cols.length], width: 1},
        hovertemplate: `class ${m + 1}: %{y:.1f}%<extra>%{text}</extra>`}));
      P.react($('#l3c'), tc, plotLayout(T, '<b>Particles per class</b> over the steps', {showlegend: true,
        xaxis: {title: {text: 'step'}}, yaxis: {title: {text: '% of particles'}, range: [0, 100]}}), {displaylogo: false, responsive: true});
    }
  };
  const name = j.name;
  const liveClasses = async () => {                 // a Class2D job's live.json: the classes as they form
    const d = await api('/api/project/live?job=' + encodeURIComponent(name));
    if (!d.classes || d.updated === liveAt || current !== name || !$('#lgal')) return;
    liveAt = d.updated;
    strip(d.round ? [['r', `Round ${d.round}${d.streaming ? ' (extraction running)' : ', final'}`]] : [['c', '2D classification']], d.round ? 'r' : 'c', d.it, d.n_iter, null);
    const rnd = d.round ? `Round <b>${d.round}</b> on ${fmt(d.particles)} particles${d.streaming ? ' (extraction still running)' : ' (all particles: final round)'} · ` : '';
    $('#lgal').innerHTML = `<div class="gal-tools"><div class="sum">${rnd}Iteration <b>${d.it}</b> of ${d.n_iter} · ${d.res} Å · ${d.changed}% changed class · ${d.classes.length} classes</div></div>
      <div class="gal" style="max-height:52vh">${d.classes.map(c => `<div class="cls" style="cursor:default"><img src="${c.images[0]}" alt="class ${c.cls}">
        <div class="lab"><span>${c.cls}</span><span>${fmt(c.n)} · ${(100 * c.share).toFixed(1)}%</span></div></div>`).join('')}</div>`;
  };
  const tick = async () => {
    clearTimeout(liveTimer);
    if (current !== name || !$('#ltail')) return;
    if (t0) { const s = Math.max(0, Date.now() / 1000 - skew - t0); $('#lel').textContent = `running for ${s >= 3600 ? Math.floor(s / 3600) + ' h ' : ''}${Math.floor(s % 3600 / 60)} min ${Math.floor(s % 60)} s`; }
    try {
      if (j.type === 'Class2D') await liveClasses();
      if (LIVE3D.includes(j.type)) await live3d();
      const d = await api(`/api/project/log?job=${encodeURIComponent(name)}&offset=${offset}`);
      if (d.text) {
        text = (text + d.text).split('\n').slice(-400).join('\n'); offset = d.offset;
        const lines = text.trimEnd().split('\n');
        $('#ltail').textContent = lines.slice(-18).join('\n');
        $('#llast').textContent = lines[lines.length - 1].replace(/^\S+ \S+ /, '');
      }
      if (n++ % 2 === 0) {                     // every ~6 s: new figures
        const jj = await api('/api/project/job?job=' + encodeURIComponent(name));
        if (current !== name) return;
        if (jj.status !== 'running') { await load(); return showJob(name, false); }
        const figs = jj.files.filter(f => /\.png$/.test(f.path) && !/\.(light|dark)\.png$/.test(f.path)).sort((a, b) => b.mtime - a.mtime);
        const mv = (jj.movie_files || []).slice(-1)[0];
        const newest = figs[0] ? figs[0].path + '?' + figs[0].mtime : mv || '';
        if (newest && newest !== lastFig) {
          lastFig = newest;
          const path = newest.split('?')[0];
          $('#lbody').classList.remove('nofig');
          $('#lfig').innerHTML = `<img src="${file(path)}&t=${Date.now()}" alt="newest figure"><figcaption>${esc(path.split('/').pop().replace(/^j\d+_/, '').replace('.png', '').replace(/_/g, ' '))}</figcaption>`;
          if (tab === 'plots' && figs.length !== $('#pane').querySelectorAll('.fig').length) { j.files = jj.files; renderPane(j); }
        }
      }
    } catch (e) { /* the next tick tries again */ }
    liveTimer = setTimeout(tick, 3000);
  };
  tick();
}
// movie jobs, while they run and after: counters, one micrograph (render, power spectrum vs CTF, its numbers) and every
// micrograph's numbers as charts; click a point to open that micrograph (cryoSPARC Live-like)
function stageMicPanel(st, j) {
  st.className = 'stage';
  const name = j.name, pickMode = j.type === 'Pick';     // a Pick job: its picks; MotionCtf: motion and CTF
  let running = j.status === 'running', fam = null;
  st.innerHTML = `<div class="mp">
    <div class="lhead"><span id="mpstate"></span> <span id="mpn" class="muted"></span>
      <span id="llast" class="muted lastline"></span></div>
    <div class="mpkpi" id="mpkpi"></div>
    <div class="mpview">
      <figure class="mpmic"><canvas id="mpc" aria-label="micrograph"></canvas><figcaption id="mpcap"></figcaption></figure>
      ${pickMode ? '' : `<figure class="mpps"><img id="mpps" alt="power spectrum and CTF fit"><figcaption>spectrum (left) · fitted CTF (right) · ring: fit resolution</figcaption></figure>`}
      <div class="mpside">
        <div class="mpnav"><button id="mpp" aria-label="previous micrograph">‹</button><input type="range" id="mpr" min="1" max="1" value="1" aria-label="micrograph"><button id="mpx" aria-label="next micrograph">›</button></div>
        <label class="tog"><input type="checkbox" id="mpf" ${running ? 'checked' : ''}> Follow the newest micrograph</label>
        <div class="lp"><label for="mplp">Low-pass filter</label><input type="range" id="mplp" min="0" max="60" step="1"><span id="mplpv"></span></div>
        <label class="tog" id="mpkl" style="display:none"><input type="checkbox" id="mpk" ${pickMode ? 'checked' : ''}> ${pickMode ? 'Show picks (blue: kept, grey: below the threshold)' : 'Show the picks of the Pick job'}</label>
        <table class="mpt" id="mpt"></table>
        <div id="mpdet"></div>
      </div>
    </div>
    ${pickMode ? '<div class="mplim card pad" id="pkth"></div><div class="card pad" id="pkcrops"></div><div class="mpplots" id="pkplots"><div id="pkscore" class="w2"></div><div id="pkper"></div><div id="pkdef"></div></div>'
      : '<div class="mpplots"><div id="mpmot"></div><div id="mploc"></div><div id="mpctf"></div><div id="mppz"></div></div>'}
    ${pickMode ? '' : '<div class="mplim card pad" id="mplim"></div>'}
    <div class="mpcharts" id="mpcharts"></div></div>`;
  const defs = MICSTATS.filter(([k]) => pickMode ? ['picks', 'candidates'].includes(k) : !['picks', 'candidates', 'particles'].includes(k));
  let rows = [], sel = null, lastN = -1, offset = 0;
  const med = a => { const b = a.filter(v => v != null).sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
  const fv = (v, d) => v == null ? '–' : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
  const show = (i, fromUser) => {
    if (!rows.length) return;
    i = Math.max(1, Math.min(rows.length, i));
    if (fromUser && $('#mpf')) $('#mpf').checked = i === rows.length && running;
    sel = i; const r = rows[i - 1];
    $('#mpr').max = rows.length; $('#mpr').value = i;
    loadMic(r.stem);
    if ($('#mpps')) $('#mpps').src = `/api/project/ps.png?job=${encodeURIComponent(name)}&stem=${encodeURIComponent(r.stem)}`;
    $('#mpcap').textContent = `${i} of ${rows.length} · ${r.stem}`;
    $('#mpt').innerHTML = defs.filter(([key]) => r[key] != null).map(([key, lab, u, d]) => `<tr><td>${lab}</td><td>${fv(r[key], d)} ${u}</td></tr>`).join('');
    detail(r.stem);
    const det = (j.movie_files || []).find(f => f.includes(r.stem));
    $('#mpdet').innerHTML = det ? `<a href="${file(det)}" target="_blank" rel="noopener">detailed figure of this micrograph</a>` : '';
    charts();
  };
  // the micrograph: unfiltered from the server, low-passed and contrast-stretched here (the slider), picks drawn here
  let micImg = null, micPicks = null, micDet = null, micStem = '';
  let lp = 20; try { lp = +(localStorage.getItem('celeris-lowpass') ?? 20); } catch (e) {}
  const lpText = () => { $('#mplp').value = lp; $('#mplpv').textContent = lp ? lp + ' Å' : 'off'; };
  const drawMic = () => {
    const c = $('#mpc'); if (!c || !micImg || !micImg.naturalWidth) return;
    const W = micImg.naturalWidth, H = micImg.naturalHeight, g = c.getContext('2d', {willReadFrequently: true});
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ap = micDet && micDet.apix ? micDet.apix : 1, bin = micDet && micDet.shape ? micDet.shape[1] / W : 1, px = ap * bin;
    const sigma = lp ? 0.1874 * lp / px : 0;                     // Gaussian: half amplitude at the low-pass resolution
    g.filter = 'none'; g.fillStyle = '#808080'; g.fillRect(0, 0, W, H);
    g.filter = sigma > 0.3 ? `blur(${sigma.toFixed(2)}px)` : 'none';
    g.drawImage(micImg, 0, 0); g.filter = 'none';
    const im = g.getImageData(0, 0, W, H), a = im.data, h = new Uint32Array(256);
    const m = Math.max(2, Math.round(Math.min(W, H) * 0.03));   // the stretch from the inside (blur darkens edges)
    for (let y = m; y < H - m; y += 2) for (let x = m; x < W - m; x += 2) h[a[(y * W + x) * 4]]++;
    let tot = h.reduce((x, y) => x + y, 0), lo = 0, hi = 255, acc = 0;
    for (let i = 0; i < 256; i++) { acc += h[i]; if (acc < tot * 0.005) lo = i; if (acc < tot * 0.995) hi = i + 1; }
    const lut = new Uint8ClampedArray(256), span = Math.max(hi - lo, 1);
    for (let i = 0; i < 256; i++) lut[i] = (i - lo) * 255 / span;
    for (let i = 0; i < a.length; i += 4) { const v = lut[a[i]]; a[i] = a[i + 1] = a[i + 2] = v; }
    g.putImageData(im, 0, 0);
    if (micPicks && micPicks.xy && micPicks.xy.length && $('#mpk') && $('#mpk').checked) {
      const r = 0.5 * (micPicks.diameter_a || 150) / px;
      g.lineWidth = Math.max(1, W / 700);
      [[0, 'rgba(160,160,160,.45)'], [1, css('--data')]].forEach(([k, col]) => {
        g.strokeStyle = col; g.beginPath();
        const kp = i => thPrev != null && micPicks.score ? (micPicks.score[i] >= thPrev ? 1 : 0) : micPicks.keep[i];
        micPicks.xy.forEach((q, i) => { if (kp(i) === k) { g.moveTo(q[0] / px + r, q[1] / px); g.arc(q[0] / px, q[1] / px, r, 0, 2 * Math.PI); } });
        g.stroke();
      });
    }
  };
  const loadMic = async stem => {
    micStem = stem;
    const im = new Image();
    im.onload = () => { if (micStem === stem) { micImg = im; drawMic(); } };
    im.src = `/api/project/micraw.png?job=${encodeURIComponent(name)}&stem=${encodeURIComponent(stem)}`;
    micPicks = null;
    if (fam && fam.pick) api(`/api/project/micpicks?job=${encodeURIComponent(name)}&stem=${encodeURIComponent(stem)}`)
      .then(d => { if (micStem === stem) { micPicks = d; drawMic(); } }).catch(() => {});
  };
  let detailFor = '';
  const detail = async stem => {                    // this micrograph's 1D CTF fit and motion (drawn here, data from the server)
    detailFor = stem;
    let d;
    try { d = await api(`/api/project/micdetail?job=${encodeURIComponent(name)}&stem=${encodeURIComponent(stem)}`); } catch (e) { return; }
    if (detailFor !== stem) return;
    micDet = d; drawMic();
    if (!$('#mpctf')) return;
    const T = THEME(), P = await CelerisPlots.plotly();
    const base = {paper_bgcolor: T.surface, plot_bgcolor: T.surface, font: {family: T.font, size: 11, color: T.ink2}, height: 250,
      margin: {l: 48, r: 14, t: 30, b: 40}, showlegend: true, legend: {orientation: 'h', x: 1, xanchor: 'right', y: 1.12, font: {size: 11}},
      hoverlabel: {bgcolor: T.well, bordercolor: T.line, font: {color: T.ink, family: T.font}}};
    const ax = {gridcolor: T.line, zeroline: false, linecolor: T.axis};
    const k = d.res_a.map(x => 1 / x), keep = k.map(x => x >= 1 / 40);
    const sel = a => a.filter((_, i) => keep[i]);
    const ticks = [20, 10, 7, 5, 4, 3, 2.5].filter(x => x >= d.res_a[d.res_a.length - 1]);
    const tr = [{type: 'scatter', mode: 'lines', name: 'spectrum', x: sel(k), y: sel(d.measured), line: {color: T.ink2, width: 1.2},
                 customdata: sel(d.res_a), hovertemplate: '%{customdata:.2f} Å<extra></extra>'}];
    if (d.fit) tr.push({type: 'scatter', mode: 'lines', name: 'fitted CTF', x: sel(k), y: sel(d.fit), line: {color: css('--data'), width: 1.6}, hoverinfo: 'skip'});
    const shapes = d.fit_res_a ? [{type: 'line', x0: 1 / d.fit_res_a, x1: 1 / d.fit_res_a, yref: 'paper', y0: 0, y1: 1, line: {color: css('--you'), width: 1, dash: 'dot'}}] : [];
    P.react($('#mpctf'), tr, Object.assign({}, base, {shapes,
      title: {text: `<b>CTF fit</b> · defocus ${fmt(Math.round((d.defocus_u + d.defocus_v) / 2))} Å · fit ${d.fit_res_a ? d.fit_res_a.toFixed(2) + ' Å' : '–'}`, x: 0.01, xanchor: 'left', font: {size: 12, color: T.ink}},
      xaxis: Object.assign({title: {text: 'resolution (Å)', font: {size: 11}}, tickvals: ticks.map(x => 1 / x), ticktext: ticks.map(String)}, ax),
      yaxis: Object.assign({showticklabels: false}, ax)}), {displaylogo: false, responsive: true});
    const note = (id, t) => { $(id).innerHTML = `<div class="pnote"><b>${t[0]}</b><span>${t[1]}</span></div>`; };
    const titled = t => ({text: t, x: 0.01, xanchor: 'left', font: {size: 12, color: T.ink}});
    const sh = d.shifts || [];
    if (sh.length > 1) {                                   // global: the whole frame's path, frame 1 marked
      const x0 = sh[0][0], y0 = sh[0][1];
      P.react($('#mpmot'), [{type: 'scatter', mode: 'lines+markers', x: sh.map(q => q[0] - x0), y: sh.map(q => q[1] - y0),
          marker: {color: sh.map((_, i) => i), colorscale: [[0, css('--data-dim')], [1, css('--data')]], size: 6}, line: {color: T.ink3, width: 1},
          customdata: sh.map((_, i) => i + 1), hovertemplate: 'frame %{customdata}: %{x:.2f}, %{y:.2f} Å<extra></extra>'}],
        Object.assign({}, base, {showlegend: false, title: titled(`<b>Global motion</b> · ${sh.length} frames · ${fv(d.drift_a, 1)} Å path`),
          xaxis: Object.assign({title: {text: 'x (Å)', font: {size: 11}}}, ax), yaxis: Object.assign({title: {text: 'y (Å)', font: {size: 11}}, scaleanchor: 'x'}, ax)}),
        {displaylogo: false, responsive: true});
    } else note('#mpmot', ['Motion', 'A single-frame micrograph: no frames to align.']);
    $('#mploc').style.display = sh.length > 1 ? '' : 'none';     // single frames: one note for both
    const L = d.local;
    if (L && L.model && L.centres) {                       // local: each patch's path beyond the global one, exaggerated
      const mx = Math.max(1e-6, ...L.model.flat().map(q => Math.hypot(q[0], q[1])));
      const k = 0.16 / mx, tr = [];
      L.model.forEach((path, p) => {
        const [cx, cy] = L.centres[p];
        tr.push({type: 'scatter', mode: 'lines', x: path.map(q => cx + k * q[0]), y: path.map(q => cy + k * q[1]), line: {color: css('--data'), width: 1.6}, hoverinfo: 'skip'});
        if (L.measured) tr.push({type: 'scatter', mode: 'markers', x: L.measured[p].map(q => cx + k * q[0]), y: L.measured[p].map(q => cy + k * q[1]),
          marker: {size: 3, color: T.ink3}, hoverinfo: 'skip'});
        tr.push({type: 'scatter', mode: 'markers', x: [cx], y: [cy], marker: {size: 5, color: T.ink2}, hovertemplate: `patch ${p + 1}: up to ${Math.max(...path.map(q => Math.hypot(q[0], q[1]))).toFixed(2)} Å<extra></extra>`});
      });
      P.react($('#mploc'), tr, Object.assign({}, base, {showlegend: false, title: titled(`<b>Local motion</b> · rms ${fv(d.local_rms_a, 2)} Å · paths ×${(k * 1000 / 1000 * (d.shape ? d.shape[1] * d.apix / 2 : 1)).toFixed(0)}`),
        xaxis: Object.assign({visible: false, range: [-1.15, 1.15]}, ax), yaxis: Object.assign({visible: false, range: [1.15, -1.15], scaleanchor: 'x'}, ax)}),
        {displaylogo: false, responsive: true});
    } else note('#mploc', ['Local motion', sh.length <= 1 ? 'Needs a movie with several frames.'
      : L && L.rms_a != null ? `rms ${fv(L.rms_a, 2)} Å. The patch paths are kept by MotionCtf jobs run from now on; this one has only the rms.`
      : 'No patch motion measured (patches = 0).']);
    const pz = d.ctf_patch;
    if (pz && pz.regions_a && pz.offsets_a) {             // patch defocus: each region's offset from the micrograph's fit
      const lim = Math.max(30, ...pz.offsets_a.map(Math.abs));
      P.react($('#mppz'), [{type: 'scatter', mode: 'markers+text', x: pz.regions_a.map(q => q[0]), y: pz.regions_a.map(q => q[1]),
          text: pz.offsets_a.map(v => (v > 0 ? '+' : '') + Math.round(v)), textfont: {size: 11, color: T.ink},
          marker: {symbol: 'square', size: 40, color: pz.offsets_a, cmin: -lim, cmax: lim, colorscale: [[0, '#2c7bb6'], [0.5, '#3a3d48'], [1, '#d7191c']]},
          hovertemplate: 'region at %{x:.0f}, %{y:.0f} Å: %{text} Å<extra></extra>'}],
        Object.assign({}, base, {showlegend: false, title: titled(`<b>Patch defocus</b> · tilt ${fv(pz.tilt_deg, 1)}° · ${pz.applied ? 'applied' : pz.significant ? 'significant, not applied' : 'not significant'}`),
          xaxis: Object.assign({visible: false}, ax), yaxis: Object.assign({visible: false, autorange: 'reversed', scaleanchor: 'x'}, ax)}),
        {displaylogo: false, responsive: true});
    } else note('#mppz', ['Patch defocus', 'Not measured for this micrograph.']);
  };
  // limits for curation (Curate job settings): points outside are orange, and Curate with these limits makes the job
  const LIM = {ctf_fit_a: [null, 8, 'max_ctf_fit_a'], defocus_a: [3000, 40000, 'min_defocus_a', 'max_defocus_a'], astigmatism_a: [null, 2000, 'max_astigmatism_a']};
  try { Object.assign(LIM, JSON.parse(localStorage.getItem('celeris-limits') || '{}')); } catch (e) {}
  const outside = r => Object.entries(LIM).some(([k, [lo, hi]]) => r[k] != null && ((lo != null && r[k] < lo) || (hi != null && r[k] > hi)));
  const limitsBar = () => {
    const lb = $('#mplim'); if (!lb || pickMode) return;
    const out = rows.filter(outside).length;
    lb.innerHTML = `<b>Limits</b>
      <label>CTF fit worse than <input type="number" step="0.5" data-l="ctf_fit_a" data-i="1" value="${LIM.ctf_fit_a[1]}"> Å</label>
      <label>defocus <input type="number" step="0.1" data-l="defocus_a" data-i="0" data-s="1e4" value="${LIM.defocus_a[0] / 1e4}"> – <input type="number" step="0.1" data-l="defocus_a" data-i="1" data-s="1e4" value="${LIM.defocus_a[1] / 1e4}"> µm</label>
      <label>astigmatism over <input type="number" step="100" data-l="astigmatism_a" data-i="1" value="${LIM.astigmatism_a[1]}"> Å</label>
      <span class="${out ? 'warnt' : 'muted'}">${out.toLocaleString()} of ${rows.length.toLocaleString()} outside</span>
      <button class="go" id="mpcur" ${fam && fam.mc ? '' : 'disabled'} title="a Curate micrographs job with these limits: Pick and Extract can then take only the micrographs it accepts">Curate with these limits</button>`;
    lb.querySelectorAll('input').forEach(i => i.onchange = () => {
      LIM[i.dataset.l][+i.dataset.i] = +i.value * (+i.dataset.s || 1);
      try { localStorage.setItem('celeris-limits', JSON.stringify(LIM)); } catch (e) {}
      limitsBar(); charts(); kpis({total: lastTotal});
    });
    $('#mpcur').onclick = async () => {
      const mcj = jobs.find(x => x.name === fam.mc.name); if (!mcj) return;
      const st = {max_ctf_fit_a: LIM.ctf_fit_a[1], min_defocus_a: LIM.defocus_a[0], max_defocus_a: LIM.defocus_a[1], max_astigmatism_a: LIM.astigmatism_a[1]};
      try { const d = await api('/api/project/new', {type: 'Curate', inputs: {micrographs: mcj.id}, settings: st, label: ''}); await load(); showJob(d.job); } catch (e) { alert(e.message); }
    };
  };
  let lastTotal = null;
  const spark = sr => {                               // the trend over the micrographs: medians of 40 blocks
    if (!sr || sr.length < 8) return '';
    const nb = Math.min(40, sr.length), pts = [];
    for (let b = 0; b < nb; b++) { const blk = sr.slice(Math.floor(b * sr.length / nb), Math.floor((b + 1) * sr.length / nb)); pts.push(med(blk)); }
    const lo = Math.min(...pts), hi = Math.max(...pts), sp = hi - lo || 1;
    return `<svg viewBox="0 0 200 30" class="spark" aria-hidden="true"><path d="${pts.map((p, i) => (i ? 'L' : 'M') + (i * 200 / (nb - 1)).toFixed(1) + ' ' + (27 - 24 * (p - lo) / sp).toFixed(1)).join('')}"/></svg>`;
  };
  const charts = async () => {
    const box = $('#mpcharts'); if (!box || !rows.length) return;
    const T = THEME(), have = defs.filter(([key]) => new Set(rows.map(r => r[key]).filter(v => v != null)).size > 1);   // not constant
    const cols = 3, nr = Math.ceil(have.length / cols), traces = [], lay = {
      height: 200 * nr + 20, margin: {l: 52, r: 12, t: 28, b: 24}, paper_bgcolor: T.surface, plot_bgcolor: T.surface, showlegend: false,
      font: {family: T.font, size: 11, color: T.ink2}, hovermode: 'closest', annotations: [],
      hoverlabel: {bgcolor: T.well, bordercolor: T.line, font: {color: T.ink, family: T.font}}};
    have.forEach(([key, lab, u, d], n) => {
      const c = n % cols, r = Math.floor(n / cols), ax = n * 2 + 1, xs = 'x' + (ax > 1 ? ax : ''), ys = 'y' + (ax > 1 ? ax : ''), xh = 'x' + (ax + 1), yh = 'y' + (ax + 1);
      const x0 = c / cols + 0.045, x1 = (c + 1) / cols - 0.075, y1 = 1 - r / nr - 0.13 / nr, y0 = 1 - (r + 1) / nr + 0.09 / nr;
      const pts = rows.filter(q => q[key] != null);
      const axis = {gridcolor: T.line, zeroline: false, linecolor: T.axis, tickfont: {size: 10}};
      lay['xaxis' + (ax > 1 ? ax : '')] = Object.assign({domain: [x0, x1], anchor: ys}, axis);
      lay['yaxis' + (ax > 1 ? ax : '')] = Object.assign({domain: [y0, y1], anchor: xs}, axis);
      lay['xaxis' + (ax + 1)] = Object.assign({domain: [x1 + 0.005, (c + 1) / cols - 0.01], anchor: yh, showticklabels: false, showgrid: false}, axis);
      lay['yaxis' + (ax + 1)] = Object.assign({domain: [y0, y1], anchor: xh, matches: ys, showticklabels: false}, axis);
      const m = med(pts.map(q => q[key]));
      lay.annotations.push({text: `<b>${lab}</b>${u ? ' (' + u + ')' : ''} · median ${fv(m, d)}`, xref: 'paper', yref: 'paper', x: x0, y: y1, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: {size: 12, color: T.ink}});
      const lim = pickMode ? null : LIM[key], bad = q => lim && ((lim[0] != null && q[key] < lim[0]) || (lim[1] != null && q[key] > lim[1]));
      if (lim) lim.slice(0, 2).forEach(v => { if (v != null) traces.push({type: 'scatter', mode: 'lines', x: [1, rows.length], y: [v, v], xaxis: xs, yaxis: ys, hoverinfo: 'skip', line: {color: css('--serious'), width: 1, dash: 'dash'}}); });
      traces.push({type: pts.length > 2000 ? 'scattergl' : 'scatter', mode: 'markers', x: pts.map(q => q.i), y: pts.map(q => q[key]), xaxis: xs, yaxis: ys,
        marker: {color: pts.map(q => bad(q) ? css('--serious') : css('--data')), size: 5, opacity: 0.85}, customdata: pts.map(q => q.i), text: pts.map(q => q.stem),
        hovertemplate: `%{text}<br>${lab} %{y:.${d}f} ${u}<extra></extra>`});
      const s = rows[(sel || 1) - 1];
      if (s && s[key] != null) traces.push({type: 'scatter', mode: 'markers', x: [s.i], y: [s[key]], xaxis: xs, yaxis: ys, hoverinfo: 'skip',
        marker: {color: css('--you'), size: 11, line: {color: T.surface, width: 2}}});
      traces.push({type: 'histogram', y: pts.map(q => q[key]), xaxis: xh, yaxis: yh, marker: {color: css('--data-dim')}, hoverinfo: 'skip', nbinsy: 30});
      if (m != null) traces.push({type: 'scatter', mode: 'lines', x: [1, rows.length], y: [m, m], xaxis: xs, yaxis: ys, hoverinfo: 'skip', line: {color: T.ink3, width: 1, dash: 'dot'}});
    });
    const P = await CelerisPlots.plotly();
    if (!$('#mpcharts')) return;
    await P.react(box, traces, lay, {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines']});
    if (!box._wired) { box._wired = true; box.on('plotly_click', e => { const p = e.points.find(q => q.customdata); if (p) show(p.customdata, true); }); }
  };
  let pk = null, thPrev = null;                      // a Pick job's own numbers (jobview.pick_stats); a threshold to try
  let cropsAt = '';
  const crops = async () => {                        // the hardest calls: candidates nearest the threshold
    const box = $('#pkcrops'); if (!box) return;
    const key = String(thPrev); if (key === cropsAt) return; cropsAt = key;
    box.innerHTML = '<b>Picks around the threshold</b> <span class="muted">reading micrographs…</span>';
    let d; try { d = await api(`/api/project/pickcrops?job=${encodeURIComponent(name)}${thPrev != null ? '&threshold=' + thPrev : ''}`); } catch (e) { box.innerHTML = ''; return; }
    if (!$('#pkcrops') || cropsAt !== key) return;
    box.innerHTML = `<div class="hh"><b>Picks around the threshold</b><span class="muted">the hardest calls, from a few micrographs: blue kept, grey left out. If the grey ones look like particles, lower the threshold.</span></div>
      <div class="pkc">${d.tiles.map(t => `<figure class="${t.kept ? 'on' : ''}" title="${esc(t.stem)}"><img src="${t.src}" alt=""><figcaption>${t.score.toFixed(3)}</figcaption></figure>`).join('')}</div>`;
  };
  const thBar = () => {
    const b = $('#pkth'); if (!b || !pk || !pk.score || !pk.score.length) return;
    const lo = Math.min(...pk.score), hi = Math.max(...pk.score), ths = (pk.rows || []).map(r => r.threshold).filter(x => x != null);
    const t0 = thPrev ?? (ths.length ? med(ths) : lo), auto = ths.length > 1 && Math.min(...ths) !== Math.max(...ths);
    const kept = thPrev == null ? pk.kept / pk.candidates : pk.score.filter(x => x >= thPrev).length / pk.score.length;
    b.innerHTML = `<b>Threshold</b><input type="range" id="pkr" min="${lo}" max="${hi}" step="${((hi - lo) / 400).toPrecision(2)}" value="${t0}" style="flex:1;min-width:200px;accent-color:var(--you)">
      <b id="pkv">${thPrev == null && auto ? 'automatic' : t0.toFixed(3)}</b><span class="muted" id="pke">keeps ${Math.round(kept * pk.candidates).toLocaleString()} of ${pk.candidates.toLocaleString()} (${(100 * kept).toFixed(0)}%)${auto ? ` · now per micrograph, median ${med(ths).toFixed(3)}` : ''}</span>
      <button class="you" id="pkapply" title="a new Pick job with this threshold on the same candidates: no picking again, takes seconds">Apply to all micrographs</button>`;
    $('#pkr').onchange = () => crops();
    $('#pkr').oninput = e => {
      thPrev = +e.target.value; $('#pkv').textContent = thPrev.toFixed(3);
      const f = pk.score.filter(x => x >= thPrev).length / pk.score.length;
      $('#pke').textContent = `keeps ~${Math.round(f * pk.candidates).toLocaleString()} of ${pk.candidates.toLocaleString()} (${(100 * f).toFixed(0)}%)`;
      drawMic(); pickPlots();
    };
    $('#pkapply').onclick = async () => {
      $('#pkapply').disabled = true;
      try { const d = await api('/api/project/rethreshold', {job: name, threshold: thPrev}); await load(); showJob(d.job); } catch (e) { alert(e.message); $('#pkapply').disabled = false; }
    };
  };
  const pickPlots = async () => {
    if (!pk || !$('#pkscore')) return;
    const P = await CelerisPlots.plotly(), T = THEME(), cfg = {displaylogo: false, responsive: true};
    if (pk.score && pk.score.length) {
      const isK = (x, i) => thPrev != null ? x >= thPrev : pk.keep[i];
      const kept = pk.score.filter(isK), rej = pk.score.filter((x, i) => !isK(x, i));
      const ths = rows.map(r => r.threshold).filter(x => x != null);
      P.react($('#pkscore'), [
        {type: 'histogram', x: rej, name: 'below the threshold', marker: {color: T.ink3}, opacity: 0.7, xbins: {size: 0.01}},
        {type: 'histogram', x: kept, name: 'kept', marker: {color: css('--data')}, opacity: 0.85, xbins: {size: 0.01}}],
        plotLayout(T, '<b>Pick scores</b> · every candidate (sampled)', {barmode: 'overlay', xaxis: {title: {text: 'score'}}, yaxis: {title: {text: 'candidates'}},
          shapes: thPrev != null ? [{type: 'line', x0: thPrev, x1: thPrev, yref: 'paper', y0: 0, y1: 1, line: {color: css('--you'), width: 2}}] : []}), cfg);
    }
    const n = rows.map(r => r.picks).filter(x => x != null);
    P.react($('#pkper'), [{type: 'histogram', x: n, marker: {color: css('--data')}, nbinsx: 40}],
      plotLayout(T, `<b>Picks per micrograph</b> · median ${fv(med(n), 0)}`, {showlegend: false, xaxis: {title: {text: 'picks'}}, yaxis: {title: {text: 'micrographs'}}}), cfg);
    const dd = rows.filter(r => r.picks != null && r.defocus_a != null);
    P.react($('#pkdef'), [{type: 'scatter', mode: 'markers', x: dd.map(r => r.defocus_a / 1e4), y: dd.map(r => r.picks), text: dd.map(r => r.stem),
        customdata: dd.map(r => r.i), marker: {color: css('--data'), size: 5, opacity: 0.75}, hovertemplate: '%{text}<br>%{x:.2f} µm: %{y} picks<extra></extra>'}],
      plotLayout(T, '<b>Picks against defocus</b>', {showlegend: false, xaxis: {title: {text: 'defocus (µm)'}}, yaxis: {title: {text: 'picks'}}}), cfg);
    if (!$('#pkdef')._wired) { $('#pkdef')._wired = true; $('#pkdef').on('plotly_click', e => { const q = e.points[0]; if (q && q.customdata) show(q.customdata, true); }); }
  };
  const kpis = d => {
    const n = rows.length, k = [[`${n.toLocaleString()}${d.total ? ' / ' + d.total.toLocaleString() : ''}`, 'micrographs']];
    if (n > 1 && running) { const dt = (rows[n - 1].t - rows[0].t) / (n - 1); if (dt > 0) k.push([`${(60 / dt).toFixed(1)}`, 'per minute' + (d.total ? ` · ~${Math.ceil((d.total - n) * dt / 60)} min left` : '')]); }
    const m = key => med(rows.map(r => r[key]));
    const sum = key => rows.reduce((a, r) => a + (r[key] || 0), 0);
    if (pickMode) {
      const P = sum('picks'), C = sum('candidates');
      k.push([P.toLocaleString(), 'particles picked'], [fv(m('picks'), 0), 'median per micrograph'],
        [`${C ? Math.round(100 * P / C) : 0}%`, `of ${C.toLocaleString()} candidates kept`]);
      if (pk) k.push([fv(pk.diameter_a, 0) + ' Å', `particle diameter · ${pk.method}`]);
      const none = rows.filter(r => r.picks === 0).length;
      if (none) k.push([none.toLocaleString(), 'micrographs without picks']);
      $('#mpkpi').innerHTML = k.map(([v, c, w, sr]) => `<div class="kpi ${w ? 'warnk' : ''}"><div class="v">${v}</div><div class="c">${esc(c)}</div>${spark(sr)}</div>`).join('');
      return;
    }
    const ser = key => rows.map(r => r[key]).filter(v => v != null);
    k.push([fv(m('defocus_a') / 1e4, 2) + ' µm', 'median defocus', false, ser('defocus_a')], [fv(m('ctf_fit_a'), 2) + ' Å', 'median CTF fit', false, ser('ctf_fit_a')],
      [fv(m('astigmatism_a'), 0) + ' Å', 'median astigmatism', false, ser('astigmatism_a')]);
    const good = rows.filter(r => r.ctf_fit_a != null && r.ctf_fit_a <= 5).length, out = rows.filter(outside).length;
    k.push([`${n ? Math.round(100 * good / n) : 0}%`, 'CTF fit ≤ 5 Å']);
    k.push([out.toLocaleString(), out ? 'outside the limits below' : 'all within the limits', out > 0]);
    if (rows.some(r => r.frames > 1)) k.push([fv(m('drift_a'), 1) + ' Å', 'median drift', false, ser('drift_a')]);
    $('#mpkpi').innerHTML = k.map(([v, c, w, sr]) => `<div class="kpi ${w ? 'warnk' : ''}"><div class="v">${v}</div><div class="c">${esc(c)}</div>${spark(sr)}</div>`).join('');
  };
  const tick = async () => {
    clearTimeout(liveTimer);
    if (current !== name || !$('#mpkpi')) return;
    try {
      const d = await api('/api/project/mics?job=' + encodeURIComponent(name));
      if (current !== name) return;
      rows = d.rows; fam = d.family || {};
      const members = ['mc', 'pick', 'extract'].map(k => fam[k]).filter(Boolean);
      running = members.some(x => x.status === 'running');
      $('#mpstate').innerHTML = (running ? '<span class="dot running pulse"></span><b>Running</b> ' : '<b>Micrographs</b> ') +
        members.map(x => `<a href="#${x.name.slice(0, 3)}" class="muted" style="margin-left:6px"><span class="dot ${esc(x.status)}"></span>${esc(x.name.split('_')[1])} ${esc(x.name.slice(0, 3))}</a>`).join('');
      $('#mpkl').style.display = fam.pick ? '' : 'none';
      $('#mpn').textContent = d.total ? `· ${rows.length} of ${d.total} micrographs` : `· ${rows.length} micrographs`;
      if (rows.length !== lastN) {
        if (pickMode) { try { pk = await api('/api/project/jobview?job=' + encodeURIComponent(name)); } catch (e) {} }
        lastN = rows.length; lastTotal = d.total; kpis(d); pickPlots(); limitsBar(); thBar(); if (pickMode && !running) crops();
        if (!sel || ($('#mpf') && $('#mpf').checked)) show(rows.length); else charts();
      }
      if (running) {
        const lg = await api(`/api/project/log?job=${encodeURIComponent((fam.mc || {}).name || name)}&offset=${offset}`);
        if (lg.text) { offset = lg.offset; const ls = lg.text.trimEnd().split('\n'); $('#llast').textContent = ls[ls.length - 1].replace(/^\S+ \S+ /, ''); }
        if (d.status !== 'running' && j.status === 'running') { await load(); return showJob(name, false); }
      }
    } catch (e) { /* next tick */ }
    if (running || j.status === 'running') liveTimer = setTimeout(tick, 5000);
  };
  lpText();
  $('#mplp').oninput = e => { lp = +e.target.value; lpText(); try { localStorage.setItem('celeris-lowpass', lp); } catch (err) {} drawMic(); };
  $('#mpr').oninput = e => show(+e.target.value, true);
  $('#mpp').onclick = () => show((sel || 1) - 1, true);
  $('#mpx').onclick = () => show((sel || 1) + 1, true);
  if ($('#mpk')) $('#mpk').onchange = drawMic;
  st.tabIndex = 0;
  st.onkeydown = e => { if (e.key === 'ArrowLeft') show((sel || 1) - 1, true); if (e.key === 'ArrowRight') show((sel || 1) + 1, true); };
  tick();
}
// classes of a 2D / 3D classification: choose the ones to keep (a sticky bar says how much is kept), sort them,
// select by resolution, see the class sizes; Save selection makes the Select job
async function stageGallery(st, j) {
  st.className = 'stage';
  st.innerHTML = '<div class="launch">loading classes…</div>';
  let g;
  try { g = await api('/api/project/classes?job=' + encodeURIComponent(j.name)); }
  catch (e) { st.className = ''; st.innerHTML = ''; return; }
  if (current !== j.name) return;
  const sel = new Set(g.chosen || []);
  const editable = j.type !== 'Select' || g.waiting;
  const is3 = g.kind === '3d', hasRes = g.classes.some(c => c.res);
  const resOf = c => c.res ?? 99;
  let sortBy = 'size';
  const rs = g.classes.map(resOf).filter(r => r < 99);
  const rlo = rs.length ? Math.floor(Math.min(...rs)) : 5, rhi = rs.length ? Math.ceil(Math.max(...rs)) : 40;
  st.innerHTML = `<div class="selbar">
      <div class="sum" id="gsum"></div><div class="bar selb"><div id="gbar" style="width:0;background:var(--data)"></div></div>
      ${editable ? `<span class="seg" role="group" aria-label="sort the classes"><button data-s="size" class="on">size</button>${hasRes ? '<button data-s="res">resolution</button>' : ''}<button data-s="num">number</button></span>
      <button class="quiet" data-a="all">All</button><button class="quiet" data-a="none">None</button><button class="quiet" data-a="inv">Invert</button>
      <button class="${g.waiting ? 'you' : 'go'}" id="gsave" disabled>${g.waiting ? 'Keep these classes' : 'Save selection'}</button>` : ''}
      ${is3 && g.classes.some(c => c.map) ? `<button id="g3d">Show the class maps in 3D</button>` : ''}
    </div>
    <div class="galwrap"><div class="gal ${is3 ? 'three' : ''}" id="gal"></div>
      <aside class="galside">
        ${editable && hasRes ? `<div class="card pad"><b>Quick select</b><label class="lp" style="margin-top:8px"><span>Resolution better than</span><b id="grv"></b>
          <input type="range" id="grs" min="${rlo}" max="${rhi}" step="0.5" value="${rhi}"></label><div class="muted" style="font-size:12.5px;margin-top:6px">Selects the classes that pass; click any class to change it.</div></div>` : ''}
        <div class="card pad"><b>Class sizes</b> <span class="muted">largest first, chosen in blue</span><svg id="gsz" viewBox="0 0 240 110" style="width:100%;height:auto;margin-top:6px" role="img" aria-label="class sizes"></svg></div>
        <div class="card pad muted" style="font-size:12.5px">${editable ? 'Saving makes a Select job: Ab initio 3D, refinements and Extract take it as their particles.' : `The classes kept from ${esc(g.source)}.`}
          <label class="muted" style="display:flex;gap:8px;align-items:center;margin-top:10px">Size <input type="range" id="gsize" min="90" max="220" value="${is3 ? 90 : 120}" style="flex:1"></label></div>
      </aside></div>`;
  const total = g.classes.reduce((a, c) => a + c.n, 0), maxN = Math.max(...g.classes.map(c => c.n), 1);
  const draw = () => {
    const cs = [...g.classes].sort(sortBy === 'res' ? (a, b) => resOf(a) - resOf(b) : sortBy === 'num' ? (a, b) => a.cls - b.cls : (a, b) => b.n - a.n);
    $('#gal').innerHTML = cs.map(c => `<button class="cls" data-c="${c.cls}" aria-pressed="${sel.has(c.cls)}" title="class ${c.cls}: ${c.n.toLocaleString()} particles${c.res ? ', ' + (+c.res).toFixed(1) + ' Å' : ''}">
        ${c.images.length > 1 ? `<span class="three">${c.images.map(s => `<img src="${s}" alt="">`).join('')}</span>` : c.images.length ? `<img src="${c.images[0]}" alt="class ${c.cls}">` : '<span style="display:block;aspect-ratio:3/1"></span>'}
        <span class="lab"><span>${c.cls}</span><span>${c.share != null ? (100 * c.share).toFixed(1) + '%' : c.n.toLocaleString()}${c.res ? ' · ' + (+c.res).toFixed(c.res < 10 ? 1 : 0) + ' Å' : ''}</span></span></button>`).join('');
    if (editable) document.querySelectorAll('#gal .cls').forEach(e => e.onclick = () => { const k = +e.dataset.c; sel.has(k) ? sel.delete(k) : sel.add(k); upd(); });
    upd();
  };
  const upd = () => {
    const n = g.classes.filter(c => sel.has(c.cls)).reduce((a, c) => a + c.n, 0), pct = 100 * n / Math.max(total, 1);
    $('#gsum').innerHTML = editable ? (sel.size ? `<b>${sel.size}</b> of ${g.classes.length} classes · <b>${n.toLocaleString()}</b> particles <span class="muted">(${pct.toFixed(1)}%)</span>`
      : `Click the classes to keep · ${g.classes.length} classes, ${total.toLocaleString()} particles`) : `${sel.size} classes kept of ${esc(g.source)} · ${n.toLocaleString()} particles`;
    $('#gbar').style.width = pct.toFixed(1) + '%';
    document.querySelectorAll('#gal .cls').forEach(e => { const k = +e.dataset.c; e.classList.toggle('on', sel.has(k)); e.classList.toggle('off', sel.size > 0 && !sel.has(k)); e.setAttribute('aria-pressed', sel.has(k)); });
    const b = $('#gsave'); if (b) b.disabled = !sel.size;
    const bars = [...g.classes].sort((a, b) => b.n - a.n), w = 240 / bars.length;
    $('#gsz').innerHTML = bars.map((c, i) => `<rect x="${(i * w).toFixed(2)}" y="${(108 - 104 * c.n / maxN).toFixed(1)}" width="${Math.max(w - 0.8, 0.8).toFixed(2)}" height="${(104 * c.n / maxN).toFixed(1)}" fill="${sel.has(c.cls) ? css('--data') : css('--axis')}"><title>class ${c.cls}: ${c.n.toLocaleString()}</title></rect>`).join('');
  };
  if ($('#g3d')) $('#g3d').onclick = () => openViewer(st, j, g.classes.filter(c => c.map).map(c => ({path: c.map})), null);
  $('#gsize').oninput = e => $('#gal').style.setProperty('--cell', e.target.value * (is3 ? 3 : 1) + 'px');
  if (editable) {
    document.querySelectorAll('.selbar [data-s]').forEach(b => b.onclick = () => { sortBy = b.dataset.s; document.querySelectorAll('.selbar [data-s]').forEach(x => x.classList.toggle('on', x === b)); draw(); });
    document.querySelectorAll('.selbar [data-a]').forEach(b => b.onclick = () => {
      const a = b.dataset.a;
      g.classes.forEach(c => { if (a === 'all') sel.add(c.cls); else if (a === 'none') sel.delete(c.cls); else sel.has(c.cls) ? sel.delete(c.cls) : sel.add(c.cls); });
      upd();
    });
    if ($('#grs')) {
      const rv = () => $('#grv').textContent = (+$('#grs').value).toFixed(1) + ' Å';
      $('#grs').oninput = () => { rv(); sel.clear(); g.classes.forEach(c => { if (resOf(c) <= +$('#grs').value) sel.add(c.cls); }); upd(); };
      rv();
    }
    $('#gsave').onclick = async () => {
      $('#gsave').disabled = true;
      try { const d = await api('/api/project/select', {job: j.name, classes: [...sel].sort((a, b) => a - b)}); await load(); showJob(d.job); }
      catch (e) { alert(e.message); $('#gsave').disabled = false; }
    };
  }
  draw();
}
function stageVolume(st, j, maps) {
  st.className = 'stage';
  const rank = p => ['_map.mrc', 'sharpened_masked.mrc', 'sharpened.mrc', 'map_nomask.mrc'].findIndex(x => p.endsWith(x));
  maps.sort((a, b) => (rank(a.path) < 0 ? 9 : rank(a.path)) - (rank(b.path) < 0 ? 9 : rank(b.path)));
  const localres = j.files.find(f => /localres\.mrc$/.test(f.path));
  const preview = j.files.find(f => /map_views\.png$/.test(f.path)) || j.files.find(f => /postprocess\.png$/.test(f.path));
  // the map itself first (the viewer opens at once), then its projections, central sections and Fourier sections
  st.innerHTML = `<div id="v3host" class="v3host"></div>
    <div class="vstrip"><div class="vrow" id="mviews">${Array.from({length: 9}, () => '<figure><div class="tile"></div></figure>').join('')}</div>
      <div class="fnote muted" id="mvname"></div></div>`;
  openViewer($('#v3host'), j, maps, localres);
  api('/api/project/mapviews?path=' + encodeURIComponent(maps[0].path)).then(d => {
    if (!$('#mviews')) return;
    $('#mviews').innerHTML = [...d.views, ...(d.fourier || [])].map(v => `<figure><div class="tile ${/Fourier/.test(v.label) ? 'four' : ''}"><img src="${v.src}" alt="${esc(v.label)}"></div><figcaption>${esc(v.label.replace('Fourier section', 'Fourier'))}</figcaption></figure>`).join('');
    $('#mvname').innerHTML = `${esc(d.name)} · box ${d.box[0]} at ${d.apix.toFixed(2)} Å/px.` + (d.fourier ? ` <b>Fourier sections</b> (log amplitudes, edge = Nyquist ${d.nyquist_a} Å): even, round discs mean detail in every direction; streaks, a missing wedge or a squashed disc mean preferred views.` : '');
  }).catch(() => {
    if ($('#mviews') && preview) $('#mviews').outerHTML = `<img class="fallback" src="${file(preview.path)}" alt="map preview">`;
  });
}
async function openViewer(st, j, maps, localres) {
  st.innerHTML = `<div class="v3"><div style="position:relative"><canvas id="v3c" aria-label="3D map"></canvas><div class="hint">drag to rotate · wheel to zoom · double-click to reset</div></div>
    <div class="ctl">
      <div><label for="v3q">Detail</label><select id="v3q" title="voxels across the particle's cube; the map is cropped to the particle first, so most maps are shown at their full pixel size">
        <option value="160">fast (160)</option><option value="256" selected>standard (256)</option><option value="384">full (384, large maps: slower)</option></select></div>
      <div><label for="v3m">Map</label><select id="v3m">${maps.map(m => `<option value="${esc(m.path)}">${esc(m.path.split('/').pop())}</option>`).join('')}</select></div>
      <div><label for="v3l">Threshold <span id="v3lv" class="lvl"></span></label><input type="range" id="v3l" min="0" max="1000" value="500">
        <div class="row" id="v3p" style="margin-top:6px"></div></div>
      <div><label class="tog" style="margin:0" title="small isolated pieces at the clean level (noise) are left out"><input type="checkbox" id="v3d" checked> Hide dust</label>
        <label class="tog" style="margin:6px 0 0" title="a Gaussian of one displayed voxel, for the view only: on by default for rough (sharpened) maps"><input type="checkbox" id="v3g"> Smooth <span class="muted" id="v3gs"></span></label></div>
      ${localres ? `<div><label class="tog" style="margin:0"><input type="checkbox" id="v3c2"> Colour by local resolution</label>
        <div id="v3r" style="display:none;margin-top:6px">
          <div class="ramp" id="v3ramp"></div>
          <div class="row muted" style="justify-content:space-between;font-size:11.5px;margin-top:3px"><span id="v3r0"></span><span id="v3rm"></span><span id="v3r1"></span></div>
          <div class="crange"><span class="muted">finest</span><input type="range" id="v3ca" min="0" max="1000"><span id="v3cav"></span>
            <span class="muted">coarsest</span><input type="range" id="v3cb" min="0" max="1000"><span id="v3cbv"></span></div>
          <div class="row" style="margin-top:6px"><select id="v3cm" aria-label="colour map"><option value="rdylbu">RdYlBu (blue = better)</option><option value="turbo">Turbo</option></select></div>
          <div class="muted" style="font-size:11.5px;margin-top:4px">from the values on the surface; outside the range the colour is clipped</div>
        </div></div>` : ''}
      <div><label>View along</label><div class="row"><button data-ax="z">z</button><button data-ax="y">y</button><button data-ax="x">x</button></div></div>
      <div class="muted" id="v3i" style="font-size:12px"></div>
      <div><button id="v3s">Save image</button></div>
    </div></div>`;
  try { viewer = CelerisVolume.create($('#v3c'), {bg: '#000000'}); }
  catch (e) { st.innerHTML = `<div class="launch err">${esc(e.message)}</div>`; return; }
  let head = null, smoothChanged = false;
  const lv = () => head ? head.min + (head.max - head.min) * $('#v3l').value / 1000 : 0;
  const setLv = v => { $('#v3l').value = Math.round(1000 * (v - head.min) / (head.max - head.min)); viewer.setLevel(v); $('#v3lv').textContent = (+v.toPrecision(3)); };
  const loadMap = async () => {
    $('#v3i').textContent = 'loading…';
    const path = $('#v3m').value;
    const sameJob = localres && path.split('/').slice(0, 2).join('/') === localres.path.split('/').slice(0, 2).join('/');
    const same = head && head.name === path.split('/').pop();
    const keep = same && !smoothChanged ? lv() : null;     // dust toggled: same map, same level
    const sm = same ? ($('#v3g').checked ? 1 : 0) : -1;     // a new map: smoothing chosen by the server
    smoothChanged = false;
    try { head = await viewer.load(`/api/project/volume?path=${encodeURIComponent(path)}&size=${$('#v3q').value}&dust=${$('#v3d').checked ? 1 : 0}&smooth=${sm}${sameJob ? '&color=' + encodeURIComponent(localres.path) : ''}`); }
    catch (e) { $('#v3i').textContent = e.message; return; }
    $('#v3p').innerHTML = Object.entries(head.levels).map(([k, v]) => `<button data-lv="${v}" title="${k === 'clean' ? 'the level where the map is one connected body (the default)' : k.endsWith('σ') ? k + ' above the noise at the box edge' : k + ' of the strongest density'}">${k}</button>`).join('');
    document.querySelectorAll('[data-lv]').forEach(b => b.onclick = () => setLv(+b.dataset.lv));
    setLv(keep ?? head.level);
    $('#v3g').checked = !!head.smooth_a; $('#v3gs').textContent = head.smooth_a ? `(${head.smooth_a} Å)` : '';
    const full = Math.abs(head.apix - head.apix_file) < 1e-3;
    $('#v3i').innerHTML = `${esc(head.name)}<br>box ${head.box} px at ${head.apix_file.toFixed(3)} Å; the particle's ${head.crop} px shown ` +
      (full ? 'at full resolution' : `at ${head.shape[0]} px, ${head.apix.toFixed(2)} Å/px`);
    if (head.color && $('#v3ca')) {
      const c = head.color, toS = x => Math.round(1000 * (x - c.lo) / Math.max(c.hi - c.lo, 1e-9)), toA = v => c.lo + (c.hi - c.lo) * v / 1000;
      $('#v3ca').value = toS(c.min); $('#v3cb').value = toS(c.max);
      const upd = () => {
        let a = toA(+$('#v3ca').value), b = toA(+$('#v3cb').value);
        if (b <= a + 0.05) { b = a + 0.05; }
        viewer.setColorRange(a, b);
        $('#v3cav').textContent = a.toFixed(1) + ' Å'; $('#v3cbv').textContent = b.toFixed(1) + ' Å';
        $('#v3r0').textContent = a.toFixed(1) + ' Å'; $('#v3rm').textContent = ((a + b) / 2).toFixed(1); $('#v3r1').textContent = b.toFixed(1) + ' Å';
      };
      $('#v3ca').oninput = upd; $('#v3cb').oninput = upd; upd();
    }
  };
  $('#v3l').oninput = () => { viewer.setLevel(lv()); $('#v3lv').textContent = +lv().toPrecision(3); };
  $('#v3m').onchange = () => { head = null; loadMap(); };
  $('#v3d').onchange = loadMap;
  $('#v3q').onchange = loadMap;
  $('#v3g').onchange = () => { smoothChanged = true; loadMap(); };
  if ($('#v3c2')) $('#v3c2').onchange = e => { viewer.setColor(e.target.checked); $('#v3r').style.display = e.target.checked ? '' : 'none'; };
  if ($('#v3cm')) $('#v3cm').onchange = e => { viewer.setColormap(e.target.value); $('#v3ramp').className = 'ramp ' + (e.target.value === 'turbo' ? 'turbo' : ''); };
  document.querySelectorAll('[data-ax]').forEach(b => b.onclick = () => viewer.view(b.dataset.ax));
  $('#v3s').onclick = () => { const a = document.createElement('a'); a.href = viewer.snapshot(); a.download = j.name + '_3d.png'; a.click(); };
  loadMap();
}
function stageMics(st, j) {
  st.className = 'stage';
  const fs = j.movie_files;
  st.innerHTML = `<div class="mic"><div class="nav"><button id="mp" aria-label="previous micrograph">‹</button>
      <input type="range" id="mr" min="0" max="${fs.length - 1}" value="0" aria-label="micrograph"><button id="mn" aria-label="next micrograph">›</button>
      <span class="muted" id="ml" style="min-width:220px;text-align:right"></span></div><img id="mi" alt="micrograph figure"></div>`;
  const show = i => { i = Math.max(0, Math.min(fs.length - 1, i)); $('#mr').value = i; $('#mi').src = file(fs[i]);
    $('#ml').textContent = `${i + 1} of ${fs.length}: ${fs[i].split('/').pop().replace(/^j\d+_/, '').replace('.png', '')}`; };
  $('#mr').oninput = e => show(+e.target.value);
  $('#mp').onclick = () => show(+$('#mr').value - 1);
  $('#mn').onclick = () => show(+$('#mr').value + 1);
  show(0);
}

// tabs
function renderPane(j) {
  const p = $('#pane');
  clearTimeout(logTimer);
  if (tab === 'plots') return panePlots(p, j);
  if (tab === 'files') {
    const fs = j.files.filter(f => !/\.(png|gif)$/.test(f.path) && !/\.plot\.json$/.test(f.path));
    p.innerHTML = fs.length ? `<div class="files">${fs.map(f => `<a href="${file(f.path)}" download>${esc(f.path)} <span class="muted">${(f.size / 1e6).toFixed(1)} MB</span></a>`).join('')}</div>` : '<p class="muted">No files besides the figures.</p>';
    return;
  }
  if (tab === 'log') { p.innerHTML = `<pre class="log" id="log"></pre>`; return followLog(j.name, 0); }
  if (tab === 'settings') {
    const hide = new Set(['history', 'figures', 'warnings', 'fsc_solvent', 'fsc_corrected']);
    const summ = Object.fromEntries(Object.entries(j.summary || {}).filter(([k, v]) => !hide.has(k) && !(Array.isArray(v) && v.length > 12)));
    p.innerHTML = `<h3 class="muted" style="margin:0 0 6px;font-size:13px">Results</h3>${kv(summ)}<h3 class="muted" style="margin:18px 0 6px;font-size:13px">Settings</h3>${kv(j.params)}`;
    return;
  }
  if (tab === 'lineage') {
    p.innerHTML = `<h3 class="muted" style="margin:0 0 6px;font-size:13px">Made from</h3>${chips(j.lineage.slice(1))}
      <h3 class="muted" style="margin:0 0 6px;font-size:13px">Used by</h3>${chips(j.dependents)}
      <button id="flowb">Draw the flowchart</button><div id="flowbox" class="flow" style="margin-top:12px"></div>`;
    document.querySelectorAll('#pane .chip').forEach(e => e.onclick = () => showJob(e.dataset.n));
    $('#flowb').onclick = async () => {
      $('#flowbox').innerHTML = '<p class="muted">drawing…</p>';
      try { const d = await api('/api/project/flow?job=' + encodeURIComponent(j.name));
        $('#flowbox').innerHTML = `<img src="${file(d.png)}&t=${Date.now()}" alt="flowchart"><p><a href="${file(d.svg)}" download>flowchart.svg</a> (text stays editable)</p>`; }
      catch (e) { $('#flowbox').innerHTML = `<p class="err">${esc(e.message)}</p>`; }
    };
  }
}
const chips = arr => arr.length ? `<div class="chips">${arr.map(n => `<span class="chip" data-n="${esc(n)}">${esc(n)}</span>`).join('')}</div>` : '<p class="muted">none</p>';
function kv(obj, depth = 0) {
  const cell = v => (v && typeof v === 'object' && !Array.isArray(v) && depth < 2) ? kv(v, depth + 1)
    : esc(Array.isArray(v) ? v.map(x => typeof x === 'object' ? JSON.stringify(x) : x).join(', ') : (typeof v === 'object' ? JSON.stringify(v) : v));
  return '<table class="kv">' + Object.entries(obj).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${cell(v)}</td></tr>`).join('') + '</table>';
}
let interactive = true;
try { interactive = localStorage.getItem('celeris-interactive') !== '0'; } catch (e) {}
// a job's figures, by topic, in cards of one size
const FIGSECTIONS = [
  ['Resolution', /(^|_)(fsc|postprocess|sharpen|localres|nu|ctfrefine|resolution)/],
  ['Classes', /(^|_)(class|classes|gallery|distribution|selection|occupancy)/],
  ['Orientations', /(^|_)(views|angdist|reprojections|projections)$/],
  ['Map', /(^|_)(map_views|mask|mode\d|variance|what_moves|latent|recenter)/],
  ['Progress', /(^|_)(progress|history|particles|iterations|overview|frames|stats)/],
];
// two columns of equal cards; a figure much wider than tall (a strip of panels) takes the whole row
function panePlots(p, j) {
  const figs = j.files.filter(f => /\.(png|gif)$/.test(f.path) && !/\.(light|dark)\.png$/.test(f.path) && !/\/movies\//.test(f.path));
  const hasJson = f => j.files.some(x => x.path === f.path.replace(/\.png$/, '.plot.json'));
  if (!figs.length) { p.innerHTML = `<p class="muted">${j.status === 'running' ? 'No figures yet: they appear here as the job writes them.' : 'This job has no figures.'}</p>`; return; }
  const base = f => f.path.split('/').pop().replace(/^j\d+_/, '').replace(/\.(png|gif)$/, '');
  const secs = [...FIGSECTIONS.map(([n]) => [n, []]), ['More', []]];
  figs.forEach((f, i) => { const k = FIGSECTIONS.findIndex(([, re]) => re.test(base(f))); secs[k < 0 ? secs.length - 1 : k][1].push(i); });
  const used = secs.filter(([, is]) => is.length);
  p.innerHTML = used.map(([n, is]) => `<section class="figsec">${used.length > 1 ? `<h3>${esc(n)}</h3>` : ''}<div class="figs">${is.map(i => `
      <div class="fig" id="card${i}"><div class="bar"><span>${esc(base(figs[i]).replace(/_/g, ' '))}</span>
      ${hasJson(figs[i]) ? `<span class="seg"><button data-i="${i}" data-m="live" class="${interactive ? 'on' : ''}">Interactive</button><button data-i="${i}" data-m="img" class="${interactive ? '' : 'on'}">Image</button></span>` : ''}</div>
      <div class="plot" id="fig${i}"></div></div>`).join('')}</div></section>`).join('');
  const size = (i, ratio) => { $('#card' + i).classList.toggle('wide', ratio > 1.9); };
  const draw = async (i, live) => {
    const f = figs[i], box = $('#fig' + i);
    if (live && hasJson(f)) {
      box.innerHTML = '<p class="muted" style="padding:14px">drawing…</p>';
      try {
        const spec = await (await fetch(file(f.path.replace(/\.png$/, '.plot.json')))).json();
        size(i, spec.width / spec.height); box.innerHTML = '';
        await CelerisPlots.render(box, spec, THEME());
      } catch (e) { box.innerHTML = `<img src="${file(f.path)}" alt=""><p class="muted">${esc(e.message)}</p>`; }
    } else {
      box.innerHTML = `<img loading="lazy" src="${file(f.path)}" alt="${esc(f.path)}">`;
      const im = box.querySelector('img'); im.onload = () => size(i, im.naturalWidth / im.naturalHeight);
    }
  };
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); draw(+e.target.id.slice(3), interactive); } }), {rootMargin: '300px'});
  figs.forEach((f, i) => io.observe($('#fig' + i)));
  document.querySelectorAll('.seg button[data-i]').forEach(b => b.onclick = () => {
    const live = b.dataset.m === 'live';
    interactive = live; try { localStorage.setItem('celeris-interactive', live ? '1' : '0'); } catch (e) {}
    b.parentNode.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    draw(+b.dataset.i, live);
  });
}
async function followLog(name, offset) {
  clearTimeout(logTimer);
  if (current !== name || !$('#log')) return;
  let d;
  try { d = await api(`/api/project/log?job=${encodeURIComponent(name)}&offset=${offset}`); } catch (e) { return; }
  if (current !== name || !$('#log')) return;
  const pre = $('#log'), atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  pre.textContent += d.text || (offset ? '' : 'No run log for this job.');
  if (atEnd) pre.scrollTop = pre.scrollHeight;
  if (d.status === 'running') logTimer = setTimeout(() => followLog(name, d.offset), 3000);
}

// ------------------------------------------------------------------ new job, report, open
async function loadTypes() { try { types = Object.fromEntries((await api('/api/project/jobtypes')).map(t => [t.type, t])); } catch (e) {} }
// New job: every job type by processing stage (uimeta.py: plain names, what each needs), then its form - the
// inputs, the few settings most runs change, and the rest folded away by topic
const usableJobs = () => jobs.filter(x => ['finished', 'queued', 'waiting', 'running'].includes(x.status));
function jobPicker() {
  current = null; renderList();
  const all = Object.values(types).sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
  if (!all.length) { $('#main').innerHTML = '<div class="empty err">The server did not send job types.</div>'; return; }
  const have = t => (t.inputs || []).filter(i => !i.optional).every(i => usableJobs().some(x => i.types.includes(x.type)));
  const groups = [];
  all.forEach(t => { const g = t.group || 'Other'; let e = groups.find(x => x[0] === g); if (!e) groups.push(e = [g, []]); e[1].push(t); });
  const last = jobs.length ? jobs.reduce((a, b) => (a.id > b.id ? a : b)) : null;
  const next = last ? all.filter(t => have(t) && (t.inputs || []).some(i => i.types.includes(last.type))).slice(0, 4) : all.filter(t => t.type === 'Import');
  const card = t => `<button class="jt ${have(t) ? '' : 'later'}" data-t="${esc(t.type)}">${icon(t.type)}
      <span class="jtn">${esc(t.name || t.type)}</span><span class="jtb">${esc(t.blurb || t.title || '')}</span>
      <span class="jtneed">${have(t) ? esc(t.needs || '') : 'Needs ' + esc((t.needs || '').replace(/^\w/, c => c.toLowerCase())) + ' first'}</span></button>`;
  $('#main').innerHTML = `<div class="head"><div class="title"><h1>New job</h1><div class="sub">Pick what to do next. Greyed jobs need an earlier step first.</div></div>
      <div></div><div class="acts"><input id="jtq" placeholder="Search, e.g. refine, mask, picking" style="width:280px" aria-label="search job types"></div></div>
    <div class="pane" style="padding-top:2px">
      ${next.length ? `<section class="jtg"><h3>Next${last ? ` after ${esc((types[last.type] || {}).name || last.type)} ${String(last.id).padStart(3, '0')}` : ''}</h3><div class="jts">${next.map(card).join('')}</div></section>` : ''}
      ${groups.map(([g, ts]) => `<section class="jtg"><h3>${esc(g)}</h3><div class="jts">${ts.map(card).join('')}</div></section>`).join('')}</div>`;
  document.querySelectorAll('.jt').forEach(b => b.onclick = () => jobForm(b.dataset.t));
  $('#jtq').oninput = e => {
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll('.jt').forEach(b => { const t = types[b.dataset.t]; b.style.display = !q || `${t.type} ${t.name} ${t.blurb} ${t.group}`.toLowerCase().includes(q) ? '' : 'none'; });
    document.querySelectorAll('.jtg').forEach(g => g.style.display = [...g.querySelectorAll('.jt')].some(b => b.style.display !== 'none') ? '' : 'none');
  };
  $('#jtq').focus();
}
function jobForm(type, params = {}) {
  current = null; renderList();
  const t = types[type];
  if (!t) return jobPicker();
  const usable = usableJobs();
  const opt = inp => {
    const ok = usable.filter(x => inp.types.includes(x.type)).sort((a, b) => b.id - a.id);
    const pre = params[inp.name] != null ? Number(params[inp.name]) : (ok.length ? ok[0].id : null);
    return (inp.optional ? '<option value="">none</option>' : ok.length ? '' : '<option value="">no such job yet</option>') +
      ok.map(x => `<option value="${x.id}" ${x.id === pre ? 'selected' : ''}>${esc((types[x.type] || {}).name || x.type)} ${String(x.id).padStart(3, '0')}${x.label ? ' · ' + esc(x.label) : ''}${x.status !== 'finished' ? ' (' + x.status + ')' : x.key ? ' · ' + esc(x.key) : ''}</option>`).join('');
  };
  const field = f => {
    const v = params[f.name] === undefined ? f.default : params[f.name], id = 'f_' + f.name;
    if (f.kind === 'bool') return `<label class="sw"><input type="checkbox" id="${id}" ${v ? 'checked' : ''}><span>${v ? 'on' : 'off'}</span></label>`;
    if (f.job_types && f.job_types.length) {
      const ok = usable.filter(x => f.job_types.includes(x.type)).sort((a, b) => b.id - a.id);
      const cur = typeof v === 'number' ? v : null;
      if (cur !== null || !f.free) return `<select id="${id}"><option value="0">none</option>${ok.map(x => `<option value="${x.id}" ${x.id === cur ? 'selected' : ''}>${esc((types[x.type] || {}).name || x.type)} ${String(x.id).padStart(3, '0')}${x.label ? ' · ' + esc(x.label) : ''}</option>`).join('')}</select>`;
    }
    if (f.choices && Array.isArray(v)) return `<span class="checks" id="${id}">${f.choices.map(([cv, cl]) => `<label><input type="checkbox" value="${esc(cv)}" ${v.includes(cv) ? 'checked' : ''}> ${esc(cl)}</label>`).join('')}</span>`;
    if (f.choices && !f.free) return `<select id="${id}">${f.choices.map(([cv, cl]) => `<option value="${esc(cv)}" ${String(cv) === String(v) ? 'selected' : ''}>${esc(cl)}</option>`).join('')}</select>`;
    const shown = Array.isArray(v) ? v.join(', ') : (v == null ? '' : v), num = f.kind === 'int' || f.kind === 'float';
    const list = f.choices ? `<datalist id="dl_${f.name}">${f.choices.map(([cv, cl]) => `<option value="${esc(cv)}">${esc(cl)}</option>`).join('')}</datalist>` : '';
    return `<input ${num ? `type="number" step="${f.kind === 'float' ? 'any' : '1'}"` : 'type="text"'} id="${id}" value="${esc(shown)}" ${list ? `list="dl_${f.name}"` : ''}>${list}`;
  };
  const row = f => `<div class="fr"><label for="f_${f.name}">${esc(f.label || f.name)}</label><div>${field(f)}<div class="help">${esc(f.help || '')}</div></div></div>`;
  const basic = t.settings.filter(f => (f.level || 'basic') === 'basic'), adv = t.settings.filter(f => f.level === 'advanced');
  const secs = [];
  adv.forEach(f => { const g = f.section || 'Other'; let e = secs.find(x => x[0] === g); if (!e) secs.push(e = [g, []]); e[1].push(f); });
  const changed = adv.filter(f => params[f.name] !== undefined && JSON.stringify(params[f.name]) !== JSON.stringify(f.default)).length;
  $('#main').innerHTML = `<div class="head"><div class="title"><h1 style="display:flex;gap:12px;align-items:center">${icon(type)}${esc(t.name || type)}</h1>
        <div class="sub">${esc(t.blurb || t.title)}</div></div><div></div>
      <div class="acts"><button class="quiet" id="fback">← All job types</button></div></div>
    <div class="pane form2" style="padding-top:4px">
      ${t.typical ? `<p class="muted typ">${esc(t.typical)}</p>` : ''}
      ${t.inputs.length ? `<section><h3>Input</h3>${t.inputs.map(i => `<div class="fr"><label for="in_${esc(i.name)}">${esc(i.label || i.name)}${i.optional ? '' : ' *'}</label>
        <div><select id="in_${esc(i.name)}">${opt(i)}</select><div class="help">${esc(i.help || i.doc)}</div><div class="cpick" id="cp_${esc(i.name)}"></div></div></div>`).join('')}</section>` : ''}
      ${basic.length ? `<section><h3>Settings</h3>${basic.map(row).join('')}</section>` : ''}
      ${adv.length ? `<details class="adv" ${changed ? 'open' : ''}><summary>More settings <span class="muted">${adv.length}${changed ? `, ${changed} changed from the default` : ''}</span></summary>
        ${secs.map(([g, fs]) => `<section><h4>${esc(g)}</h4>${fs.map(row).join('')}</section>`).join('')}</details>` : ''}
      <section><div class="fr"><label for="flabel">Label</label><div><input id="flabel" placeholder="optional, e.g. C8_nu" value="${esc(params.__label || '')}"><div class="help">A short name shown next to the job number.</div></div></div></section>
      <p class="frun"><button class="go" id="frun">Add to the queue</button> <span class="muted">It runs when its inputs are ready.</span> <span class="err" id="ferr"></span></p></div>`;
  document.querySelectorAll('.sw input').forEach(c => c.onchange = () => c.nextElementSibling.textContent = c.checked ? 'on' : 'off');
  $('#fback').onclick = jobPicker;
  // a multi-class job as the reference: its class maps to click (reference_class), or, for references lists
  // (Heterogeneous refinement), maps added as "job:class"
  const rc = $('#f_reference_class');
  if (rc) {
    rc.closest('.fr').style.display = 'none';
    const target = ['reference', 'poses'].find(n => $('#in_' + n));
    const watch = () => {
      const r = $('#in_reference'), q = $('#in_poses'), src = (r && r.value) ? 'reference' : 'poses';
      ['reference', 'poses'].forEach(n => { if ($('#cp_' + n)) $('#cp_' + n).innerHTML = ''; });
      const el = $('#in_' + src); if (!el || !el.value) return;
      classStrip($('#cp_' + src), el.value, [+rc.value || 0], (k, on) => { rc.value = on ? k : 0; }, true);
    };
    ['reference', 'poses'].forEach(n => { if ($('#in_' + n)) $('#in_' + n).addEventListener('change', watch); });
    if (target) watch();
  }
  const refs = $('#f_references');
  if (refs) {
    const box = document.createElement('div'); box.className = 'cpick';
    const multi = usable.filter(x => ['InitialModel', 'Refine3D', 'HeteroRefine', 'Class3D', 'LocalRefine', 'Sharpen', 'ModeMaps'].includes(x.type)).sort((a, b) => b.id - a.id);
    box.innerHTML = `<div class="cprow"><span class="muted">Add maps from</span><select id="refsrc"><option value="">a job…</option>${multi.map(x => `<option value="${x.id}">${esc((types[x.type] || {}).name || x.type)} ${String(x.id).padStart(3, '0')}${x.key ? ' · ' + esc(x.key) : ''}</option>`).join('')}</select>
      <button type="button" class="quiet" id="refdecoy" title="a junk reference: the first map's phases randomised (catches junk particles)">+ decoy</button></div><div id="refstrip"></div>`;
    refs.parentNode.insertBefore(box, refs.nextSibling);
    const list = () => refs.value.split(/[\s,]+/).filter(Boolean);
    $('#refdecoy').onclick = () => { refs.value = [...list(), 'decoy'].join(', '); };
    $('#refsrc').onchange = e => {
      const id = e.target.value; if (!id) { $('#refstrip').innerHTML = ''; return; }
      const has = () => list().filter(r => r.split(':')[0] === String(+id)).map(r => +(r.split(':')[1] || 0));
      classStrip($('#refstrip'), id, has(), (k, on) => {
        const tag = k ? `${+id}:${k}` : String(+id), cur = list().filter(r => r !== tag);
        refs.value = (on ? [...cur, tag] : cur).join(', ');
      }, false);
    };
  }
  $('#frun').onclick = async () => {
    const inputs = {}, settings = {};
    t.inputs.forEach(i => { const v = $('#in_' + i.name).value; if (v) inputs[i.name] = Number(v); });
    t.settings.forEach(f => {
      const el = $('#f_' + f.name);
      settings[f.name] = f.kind === 'bool' ? el.checked
        : el.classList.contains('checks') ? [...el.querySelectorAll('input:checked')].map(c => c.value).join(',') : el.value;
    });
    $('#frun').disabled = true;
    try { const d = await api('/api/project/new', {type, inputs, settings, label: $('#flabel').value}); await load(); showJob(d.job); }
    catch (e) { $('#ferr').textContent = e.message; $('#frun').disabled = false; }
  };
}
// the class maps of a job (projections, particles, resolution) to click: single = one class at most; a job without
// classes shows its one map (k = 0)
async function classStrip(box, jobId, chosen, onPick, single) {
  if (!box) return;
  box.innerHTML = '<span class="muted">reading the classes…</span>';
  const jj = jobs.find(x => x.id === +jobId);
  let g = null;
  try { g = await api('/api/project/classes?job=' + encodeURIComponent(jj ? jj.name : jobId)); } catch (e) { g = null; }
  if (!g || g.kind !== '3d' || !g.classes.length) { box.innerHTML = single ? '' : `<span class="muted">${esc(jj ? jj.name : jobId)} has one map.</span> <button type="button" class="quiet" id="cp1">Add it</button>`;
    if ($('#cp1')) $('#cp1').onclick = () => onPick(0, true); return; }
  const on = new Set(chosen.filter(Boolean));
  const draw = () => {
    box.innerHTML = `<div class="muted cphint">${single ? 'Start from which class? Click one (none chosen: the class most of the particles are in).' : 'Click the classes to use as references.'}</div>
      <div class="cpgrid">${g.classes.map(c => `<button type="button" class="cpc ${on.has(c.cls) ? 'on' : ''}" data-k="${c.cls}">
        <span class="three">${c.images.slice(0, 3).map(s => `<img src="${s}" alt="">`).join('')}</span>
        <span class="cpl"><b>Class ${c.cls}</b> ${c.n.toLocaleString()}${c.share != null ? ' · ' + (100 * c.share).toFixed(0) + '%' : ''}${c.res ? ' · ' + (+c.res).toFixed(1) + ' Å' : ''}</span></button>`).join('')}</div>`;
    box.querySelectorAll('.cpc').forEach(b => b.onclick = () => {
      const k = +b.dataset.k, was = on.has(k);
      if (single) on.clear();
      was ? on.delete(k) : on.add(k);
      onPick(k, !was); draw();
    });
  };
  draw();
}
$('#newjob').onclick = jobPicker;
$('#home').onclick = e => { e.preventDefault(); showHome(); };
$('#report').onclick = async () => {
  current = null; renderList();
  $('#main').innerHTML = '<div class="empty">Making the report…</div>';
  let d;
  try { d = await api('/api/project/report', {}); }
  catch (e) { $('#main').innerHTML = `<div class="empty"><h1>No final jobs yet</h1><p>${esc(e.message)}</p><p>Open the jobs whose maps or particles the publication reports and press ☆ Mark final.</p></div>`; return; }
  $('#main').innerHTML = `<div class="head"><div><h1>Report</h1><div class="sub">${d.jobs.length} jobs lead to ${d.finals.map(esc).join(', ')}; written to the project's report folder</div></div>
      <div class="acts"><a href="${file(d.flowchart_svg)}" download><button>flowchart.svg</button></a><a href="${file(d.methods)}" download><button>methods.md</button></a><a href="${file(d.settings)}" download><button>settings.csv</button></a></div></div>
    <div class="pane" style="padding-top:4px"><div class="flow"><img src="${file(d.flowchart)}&t=${Date.now()}" alt="flowchart"></div>
    <h3 class="muted" style="margin:22px 0 6px;font-size:13px">Methods</h3><div class="methods">${esc(d.methods_text)}</div></div>`;
};
document.querySelectorAll('.menu-list button').forEach(b => b.addEventListener('click', () => b.closest('details').open = false));
$('#open').onclick = async () => {
  const root = prompt('Project folder (the one with project.db). Windows paths such as E:\\linux\\celeris\\inx6 work too:', '');
  if (!root) return;
  try { await api('/api/project/open', {root}); } catch (e) { alert(e.message); return; }
  current = null; load().then(showHome);
};
$('#create').onclick = async () => {
  const root = prompt('Folder for the new project (new or empty). Windows paths such as E:\\linux\\celeris\\new work too:', '');
  if (!root) return;
  try { await api('/api/project/open', {root, create: true}); } catch (e) { alert(e.message); return; }
  current = null; await load(); jobForm('Import');
};

window.addEventListener('hashchange', () => { const h = decodeURIComponent(location.hash.slice(1)); const m = jobs.find(j => j.name.startsWith(h)); if (m && m.name !== current) showJob(m.name); });
loadTypes();
if (location.hash.length > 1) current = decodeURIComponent(location.hash.slice(1));
load().then(() => { const m = current && jobs.find(j => j.name.startsWith(current)); if (m) showJob(m.name); else showHome(); });
