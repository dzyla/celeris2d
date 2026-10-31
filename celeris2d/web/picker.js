// The ManualPick job's picker (manualpick.py, manualpick_api.py): pick by hand, train a template picker on the picks,
// check its suggestions on the next micrographs, then Pick all through the queue. Everything is saved as you go.
//   CelerisPicker.mount(el, {job, api, file, css, theme})   job: /api/project/job?job=NAME; api(url, body?) -> JSON
// Renders into el; timers and the key handler end when el leaves the page.
(function () {
'use strict';
const STYLE = `
.pk-stage { background: #000; padding: 12px 14px 14px; }
.pk-top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.pk-steps { display: flex; gap: 4px; }
.pk-step { display: flex; align-items: center; gap: 6px; padding: 4px 10px 4px 5px; border-radius: 999px; background: var(--well);
  color: var(--ink3); font-size: 12px; border: 1px solid var(--line); white-space: nowrap; }
.pk-step b { width: 18px; height: 18px; border-radius: 50%; display: inline-grid; place-items: center; font-size: 11px; background: var(--raise); color: var(--ink2); }
.pk-step.done { color: var(--ink2); } .pk-step.done b { background: var(--good); color: #fff; }
.pk-step.now { color: var(--ink); border-color: var(--you); } .pk-step.now b { background: var(--you); color: #1b1404; }
.pk-next { flex: 1 1 320px; color: var(--ink); font-size: 13.5px; line-height: 1.4; min-width: 0; min-height: 3.9em; display: flex; align-items: center; }
.pk-next > span { display: block; }
.pk-next b { color: var(--you); font-weight: 600; }
.pk-main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 14px; align-items: start; }
@media (max-width: 1100px) { .pk-main { grid-template-columns: 1fr; } }
.pk-view { position: relative; min-width: 0; }
.pk-canvas { display: block; width: 100%; background: #111; border-radius: 6px; cursor: crosshair; touch-action: none; }
.pk-canvas.hit { cursor: pointer; } .pk-canvas.drag { cursor: grabbing; }
.pk-busy { position: absolute; inset: 0; display: none; place-items: center; background: rgba(0,0,0,.45); color: var(--ink); font-size: 14px; border-radius: 6px; }
.pk-busy.on { display: grid; }
.pk-bar { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.pk-bar input[type=range] { flex: 1 1 160px; }
.pk-cap { color: var(--ink2); font-size: 12.5px; margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; align-items: baseline; }
.pk-cap .s { color: var(--ink); font-weight: 600; }
.pk-legend { display: flex; gap: 12px; color: var(--ink3); font-size: 12px; flex-wrap: wrap; }
.pk-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 50%; border: 2px solid; vertical-align: -1px; margin-right: 4px; }
.pk-side { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.pk-card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.pk-card h4 { margin: 0 0 8px; font-size: 13.5px; color: var(--ink); font-weight: 600; display: flex; justify-content: space-between; }
.pk-kpi { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.pk-kpi div { background: var(--well); border-radius: 6px; padding: 6px 8px; }
.pk-kpi .v { font-size: 18px; font-weight: 600; color: var(--ink); } .pk-kpi .c { font-size: 11px; color: var(--ink3); }
.pk-btns { display: flex; gap: 6px; flex-wrap: wrap; }
.pk-btns button { flex: 1 1 auto; }
.pk-note { color: var(--ink2); font-size: 12px; margin-top: 6px; line-height: 1.4; }
.pk-err { color: var(--bad); font-size: 12px; margin-top: 6px; }
.pk-tm { display: flex; gap: 4px; margin-top: 8px; overflow-x: auto; }
.pk-tm figure { margin: 0; text-align: center; font-size: 11px; color: var(--ink3); }
.pk-tm img { width: 64px; height: 64px; display: block; border-radius: 4px; background: #000; image-rendering: auto; }
.pk-hist { width: 100%; height: 56px; display: block; margin-top: 4px; }
.pk-th { display: flex; align-items: center; gap: 8px; } .pk-th input { flex: 1; } .pk-th span { min-width: 44px; text-align: right; color: var(--ink); font-weight: 600; }
.pk-list { max-height: 260px; overflow-y: auto; margin: 0 -4px; position: relative; }
.pk-row { display: grid; grid-template-columns: 34px minmax(0, 1fr) 54px 44px; gap: 6px; padding: 3px 4px; border-radius: 4px; cursor: pointer; font-size: 12px; color: var(--ink2); align-items: center; }
.pk-row:hover { background: var(--well); } .pk-row.on { background: var(--raise); color: var(--ink); }
.pk-row .n { color: var(--ink3); text-align: right; } .pk-row .st { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pk-chip { font-size: 10.5px; padding: 1px 6px; border-radius: 999px; text-align: center; background: var(--well); color: var(--ink3); }
.pk-chip.reviewed { background: rgba(12,163,12,.18); color: var(--good); } .pk-chip.suggested { background: var(--data-dim); color: var(--data); }
.pk-chip.stale { background: var(--you-dim); color: var(--you); } .pk-chip.output { background: var(--well); color: var(--ink2); }
.pk-filt { display: flex; gap: 4px; margin-bottom: 6px; } .pk-filt button { padding: 2px 8px; font-size: 11.5px; }
.pk-filt button.on { border-color: var(--data); color: var(--ink); }
.pk-keys { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 11.5px; color: var(--ink2); }
.pk-keys kbd { font: 11px/1.5 var(--font); background: var(--well); border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; color: var(--ink); white-space: nowrap; }
.pk-prog { height: 6px; border-radius: 3px; background: var(--well); overflow: hidden; margin-top: 6px; } .pk-prog div { height: 100%; background: var(--data); }
.pk-saved { color: var(--ink3); font-size: 11.5px; margin-left: auto; }
.pk-banner { background: var(--you-dim); color: var(--ink); border: 1px solid var(--you); border-radius: 8px; padding: 7px 12px; font-size: 12.5px; margin-bottom: 10px; }
`;

function mount(el, ctx) {
  const {api, css} = ctx;
  const job = ctx.job.name;
  const Q = s => el.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const enc = encodeURIComponent;
  const BASE = '/api/project/manualpick';
  if (!document.getElementById('pk-style')) {
    const s = document.createElement('style'); s.id = 'pk-style'; s.textContent = STYLE; document.head.appendChild(s);
  }
  el.className = 'stage pk-stage';
  el.innerHTML = `
    <div class="pk-top">
      <div class="pk-steps" id="pk-steps"></div>
      <div class="pk-next" id="pk-next">Loading…</div>
    </div>
    <div class="pk-main">
      <div class="pk-view">
        <canvas class="pk-canvas" id="pk-c" tabindex="0" aria-label="micrograph: click to pick, click a pick to remove it"></canvas>
        <div class="pk-busy" id="pk-busy"></div>
        <div class="pk-cap"><span class="s" id="pk-stem"></span><span id="pk-info"></span><span class="pk-saved" id="pk-saved"></span></div>
        <div class="pk-bar">
          <button id="pk-prev" title="previous micrograph (←)">‹ Prev</button>
          <input type="range" id="pk-nav" min="1" max="1" value="1" aria-label="micrograph">
          <button id="pk-nextm" title="next micrograph (→)">Next ›</button>
          <button id="pk-accept" class="go" title="keep the blue suggestions as picks and go to the next micrograph to check (Enter)">Accept &amp; next ⏎</button>
        </div>
        <div class="pk-bar">
          <div class="pk-legend"><span><i style="border-color:var(--you)"></i>your picks</span><span><i style="border-color:var(--data)"></i>suggested</span>
            <span><i style="border-color:var(--good)"></i>Pick all result</span><span style="color:var(--bad)">✕ removed suggestion</span></div>
          <label class="tog" style="margin:0 0 0 auto"><input type="checkbox" id="pk-show" checked> Show picks (H)</label>
          <label class="tog" style="margin:0" title="Gaussian low-pass of the displayed micrograph">Low-pass <input type="range" id="pk-lp" min="0" max="60" step="1" style="width:90px"> <span id="pk-lpv"></span></label>
        </div>
      </div>
      <div class="pk-side">
        <div class="pk-card"><h4>Picks <span id="pk-diam"></span></h4><div class="pk-kpi" id="pk-kpi"></div></div>
        <div class="pk-card"><h4>Picker</h4>
          <div class="pk-btns">
            <button id="pk-train" class="go" title="make templates from your picks, then pick the next micrographs with them (T)">Train &amp; suggest (T)</button>
            <button id="pk-more" title="pick the next micrographs with the current templates (S)">Suggest more (S)</button>
            <button id="pk-this" title="pick this micrograph with the templates (P)">This one (P)</button>
          </div>
          <div id="pk-tm"></div>
          <div id="pk-thr" style="display:none;margin-top:8px">
            <div class="pk-th"><label for="pk-ths" style="font-size:12px;color:var(--ink2)">Threshold</label><input type="range" id="pk-ths" min="0" max="1" step="0.002"><span id="pk-thv"></span></div>
            <canvas class="pk-hist" id="pk-hist" width="600" height="112" aria-label="score distribution"></canvas>
            <div class="pk-note" id="pk-thn"></div>
          </div>
          <div class="pk-err" id="pk-err"></div>
        </div>
        <div class="pk-card"><h4>Finish</h4><div id="pk-fin"></div></div>
        <div class="pk-card"><h4>Micrographs <span id="pk-lcount"></span></h4>
          <div class="pk-filt" id="pk-filt">${[['all', 'All'], ['check', 'To check'], ['reviewed', 'Reviewed'], ['new', 'Not seen']].map(([k, t]) => `<button data-f="${k}">${t}</button>`).join('')}</div>
          <div class="pk-list" id="pk-list"></div></div>
        <div class="pk-card"><h4>Keys</h4><div class="pk-keys">
          <span><kbd>click</kbd></span><span>add a pick · on a pick: remove it</span>
          <span><kbd>drag</kbd> <kbd>wheel</kbd></span><span>pan · zoom (<kbd>0</kbd> whole micrograph)</span>
          <span><kbd>←</kbd> <kbd>→</kbd></span><span>previous / next micrograph</span>
          <span><kbd>Enter</kbd></span><span>accept the suggestions, next to check</span>
          <span><kbd>T</kbd> <kbd>S</kbd> <kbd>P</kbd></span><span>train &amp; suggest · suggest more · this one</span>
          <span><kbd>[</kbd> <kbd>]</kbd></span><span>threshold down / up</span>
          <span><kbd>U</kbd> <kbd>Ctrl Z</kbd></span><span>undo on this micrograph</span>
          <span><kbd>H</kbd></span><span>hide / show picks</span></div></div>
      </div>
    </div>`;

  // ------------------------------------------------------------------ state
  let S = null;                  // /state
  let M = null;                  // /mic of the shown micrograph (+ local edits)
  let idx = 0, filter = 'all', th = null, show = true, dirty = false, saveTimer = null, alive = true, pollTimer = null;
  let undo = [];
  const key = `celeris-mp-${job}`;
  const ls = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
  let lp = +(ls('celeris-lowpass') ?? 20); if (!(lp >= 0)) lp = 20;
  const editable = () => S && !['queued', 'running'].includes(S.status);

  // ------------------------------------------------------------------ geometry: A <-> binned image pixels
  // raw_png bins by an integer factor f (pixel i holds micrograph pixels i*f .. i*f+f-1); xy_a = micrograph pixel * apix
  const toImg = (xa) => (xa / M.apix + 0.5) / M.bin;
  const toA = (u) => Math.round((u * M.bin - 0.5) * M.apix * 10) / 10;
  const rImg = () => 0.5 * S.settings.diameter_a / (M.apix * M.bin);
  const near = (a, b, d) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < d * d;

  // the suggestions shown: above the threshold, past the contamination filters, not removed, not on one of your picks
  const visibleSuggestions = () => {
    if (!M || M.state !== 'suggested' || !M.suggest) return [];
    const D = S.settings.diameter_a, rej = M.rejected, pk = M.picks, md = S.settings.min_dist_frac * D, out = [];
    const g = M.suggest;
    for (let i = 0; i < g.xy.length; i++) {
      if (!g.ok[i] || g.score[i] < th) continue;
      const q = g.xy[i];
      if (rej.some(r => near(r, q, 0.3 * D)) || pk.some(p => near(p, q, md))) continue;
      out.push(q);
    }
    return out;
  };

  // ------------------------------------------------------------------ drawing
  const cv = Q('#pk-c');
  const g = cv.getContext('2d');
  const base = document.createElement('canvas');          // the filtered, stretched micrograph at its binned size
  let img = null, view = null, hover = null;
  const process = () => {
    if (!img || !M) return;
    const W = img.naturalWidth, H = img.naturalHeight;
    base.width = W; base.height = H;
    const b = base.getContext('2d', {willReadFrequently: true});
    const px = M.apix * M.bin, sigma = lp ? 0.1874 * lp / px : 0;           // Gaussian: half amplitude at lp
    b.filter = 'none'; b.fillStyle = '#808080'; b.fillRect(0, 0, W, H);
    b.filter = sigma > 0.3 ? `blur(${sigma.toFixed(2)}px)` : 'none';
    b.drawImage(img, 0, 0); b.filter = 'none';
    const im = b.getImageData(0, 0, W, H), a = im.data, h = new Uint32Array(256);
    const m = Math.max(2, Math.round(Math.min(W, H) * 0.03));              // stretch from the inside (blur darkens edges)
    for (let y = m; y < H - m; y += 2) for (let x = m; x < W - m; x += 2) h[a[(y * W + x) * 4]]++;
    let tot = h.reduce((x, y) => x + y, 0), lo = 0, hi = 255, acc = 0;
    for (let i = 0; i < 256; i++) { acc += h[i]; if (acc < tot * 0.005) lo = i; if (acc < tot * 0.995) hi = i + 1; }
    const lut = new Uint8ClampedArray(256), span = Math.max(hi - lo, 1);
    for (let i = 0; i < 256; i++) lut[i] = (i - lo) * 255 / span;
    for (let i = 0; i < a.length; i += 4) { const v = lut[a[i]]; a[i] = a[i + 1] = a[i + 2] = v; }
    b.putImageData(im, 0, 0);
  };
  const fit = () => {                                     // canvas size: the column width, at most 80 % of the window height
    if (!img) return;
    const W = img.naturalWidth, H = img.naturalHeight;
    const cw = Q('.pk-view').clientWidth || 800, ch = Math.max(320, window.innerHeight * 0.72);
    const s = Math.min(cw / W, ch / H), dpr = window.devicePixelRatio || 1;
    cv.style.width = `${Math.round(W * s)}px`; cv.style.height = `${Math.round(H * s)}px`;
    cv.width = Math.round(W * s * dpr); cv.height = Math.round(H * s * dpr);
    if (!view || view.W !== W || view.H !== H) view = {W, H, z: 1, ox: 0, oy: 0};
    view.k = cv.width / W;                                 // canvas px per image px at zoom 1
  };
  const scale = () => view.k * view.z;
  const clampView = () => {
    const vw = view.W / view.z, vh = view.H / view.z;
    view.ox = Math.min(Math.max(view.ox, 0), view.W - vw); view.oy = Math.min(Math.max(view.oy, 0), view.H - vh);
  };
  const draw = () => {
    if (!img || !M || !view) return;
    const s = scale(), dpr = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#111'; g.fillRect(0, 0, cv.width, cv.height);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.setTransform(s, 0, 0, s, -view.ox * s, -view.oy * s);
    g.drawImage(base, 0, 0);
    if (!show) return;
    const r = rImg(), lw = Math.max(1.2, 1.6 * dpr) / s;
    const ring = (pts, col, w = lw, dash = null) => {
      if (!pts.length) return;
      g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash ? dash.map(x => x / s) : []); g.beginPath();
      pts.forEach(q => { const x = toImg(q[0]), y = toImg(q[1]); g.moveTo(x + r, y); g.arc(x, y, r, 0, 2 * Math.PI); });
      g.stroke(); g.setLineDash([]);
    };
    if (M.output && M.state !== 'reviewed' && M.state !== 'suggested') ring(M.output, css('--good'));
    ring(visibleSuggestions(), css('--data'));
    ring(M.picks.filter(p => p[2] === 0), css('--you'), lw * 1.3);
    ring(M.picks.filter(p => p[2] !== 0), css('--you'), lw * 1.3, [5, 3]);
    if (M.rejected.length) {                              // removed suggestions: small crosses
      g.strokeStyle = css('--bad'); g.lineWidth = lw; g.beginPath();
      const c = r * 0.3;
      M.rejected.forEach(q => { const x = toImg(q[0]), y = toImg(q[1]); g.moveTo(x - c, y - c); g.lineTo(x + c, y + c); g.moveTo(x + c, y - c); g.lineTo(x - c, y + c); });
      g.stroke();
    }
    if (hover) { g.strokeStyle = '#fff'; g.lineWidth = lw * 2; g.beginPath(); g.arc(toImg(hover[0]), toImg(hover[1]), r * 1.12, 0, 2 * Math.PI); g.stroke(); }
  };

  // ------------------------------------------------------------------ pointer: click to pick / remove, drag to pan, wheel to zoom
  const imgAt = ev => {
    const b = cv.getBoundingClientRect(), dpr = cv.width / b.width;
    return [view.ox + (ev.clientX - b.left) * dpr / scale(), view.oy + (ev.clientY - b.top) * dpr / scale()];
  };
  const hitAt = (u, v) => {                                // the nearest pick / suggestion under the pointer
    const r = rImg() * 0.8, cand = [];
    M.picks.forEach((p, i) => cand.push(['pick', i, p]));
    visibleSuggestions().forEach(q => cand.push(['sug', -1, q]));
    if (M.output && M.state !== 'reviewed' && M.state !== 'suggested') M.output.forEach((q, i) => cand.push(['out', i, q]));
    let best = null, bd = r * r;
    cand.forEach(c => { const d = (toImg(c[2][0]) - u) ** 2 + (toImg(c[2][1]) - v) ** 2; if (d < bd) { bd = d; best = c; } });
    return best;
  };
  const snapshot = () => { undo.push(JSON.stringify({picks: M.picks, rejected: M.rejected, state: M.state})); if (undo.length > 200) undo.shift(); };
  const edited = () => { dirty = true; Q('#pk-saved').textContent = 'saving…'; clearTimeout(saveTimer); saveTimer = setTimeout(save, 350); refreshCounts(); draw(); };
  const onClick = (u, v) => {
    if (!editable()) return err('Pick all is running: the picks can change when it has finished.');
    if (u < 0 || v < 0 || u > view.W || v > view.H) return;
    const h = hitAt(u, v);
    snapshot();
    if (M.output && M.state !== 'reviewed' && M.state !== 'suggested') {   // editing a Pick all result: it becomes yours
      M.picks = M.output.map(q => [q[0], q[1], 1]); M.output = null; M.state = 'reviewed';
      const h2 = h && h[0] === 'out' ? M.picks[h[1]] : null;
      if (h2) { M.picks.splice(h[1], 1); M.rejected.push([h2[0], h2[1]]); return edited(); }
    }
    if (h && h[0] === 'pick') {
      const [p] = M.picks.splice(h[1], 1);
      if (p[2] !== 0) M.rejected.push([p[0], p[1]]);              // a removed suggestion: a negative for the threshold
    } else if (h && h[0] === 'sug') {
      M.rejected.push([h[2][0], h[2][1]]);
    } else {
      M.picks.push([toA(u), toA(v), 0]);
      if (!M.state) M.state = 'reviewed';                          // picked by hand: kept as you pick it
    }
    hover = null; edited();
  };
  let down = null;
  cv.addEventListener('pointerdown', ev => { if (!view) return; cv.setPointerCapture(ev.pointerId); down = {x: ev.clientX, y: ev.clientY, ox: view.ox, oy: view.oy, moved: false}; });
  cv.addEventListener('pointermove', ev => {
    if (!view || !M) return;
    if (down) {
      const dx = ev.clientX - down.x, dy = ev.clientY - down.y;
      if (Math.hypot(dx, dy) > 4) down.moved = true;
      if (down.moved && view.z > 1) {
        const b = cv.getBoundingClientRect(), k = cv.width / b.width / scale();
        view.ox = down.ox - dx * k; view.oy = down.oy - dy * k; clampView(); cv.classList.add('drag'); draw();
      }
      return;
    }
    const [u, v] = imgAt(ev), h = hitAt(u, v), q = h ? h[2] : null;
    if (q !== hover) { hover = q; cv.classList.toggle('hit', !!q); draw(); }
  });
  cv.addEventListener('pointerup', ev => {
    if (!down) return;
    const moved = down.moved; down = null; cv.classList.remove('drag');
    if (!moved && ev.button === 0) onClick(...imgAt(ev));
  });
  cv.addEventListener('pointerleave', () => { if (hover) { hover = null; draw(); } });
  cv.addEventListener('wheel', ev => {
    if (!view) return;
    ev.preventDefault();
    const [u, v] = imgAt(ev), z = Math.min(8, Math.max(1, view.z * (ev.deltaY < 0 ? 1.25 : 0.8)));
    view.ox = u - (u - view.ox) * view.z / z; view.oy = v - (v - view.oy) * view.z / z; view.z = z; clampView(); draw();
  }, {passive: false});
  cv.addEventListener('contextmenu', ev => ev.preventDefault());
  const onResize = () => { fit(); draw(); };
  window.addEventListener('resize', onResize);

  // ------------------------------------------------------------------ server
  const busy = (t) => { const b = Q('#pk-busy'); b.textContent = t || ''; b.classList.toggle('on', !!t); };
  const err = (t) => { Q('#pk-err').textContent = t || ''; };
  const save = async () => {
    clearTimeout(saveTimer);
    if (!dirty || !M) return;
    dirty = false;
    const m = M;
    try {
      await api(BASE + '/picks', {job, stem: m.stem, picks: m.picks, rejected: m.rejected, state: m.state});
      if (M === m) Q('#pk-saved').textContent = 'saved';
      if (S) S.edited_after_run = S.status === 'finished';
    } catch (e) { dirty = true; Q('#pk-saved').textContent = 'not saved: ' + e.message; }
  };
  const flushBeacon = () => {                              // the page goes away: the last edits still reach the server
    if (!dirty || !M) return;
    try { navigator.sendBeacon(BASE + '/picks', new Blob([JSON.stringify({job, stem: M.stem, picks: M.picks, rejected: M.rejected, state: M.state})], {type: 'application/json'})); dirty = false; } catch (e) {}
  };
  const loadState = async () => {
    S = await api(`${BASE}/state?job=${enc(job)}`);
    if (th === null || S.threshold !== null && Math.abs((S.threshold ?? 0) - (th ?? 0)) > 1e-6 && !thDragging) th = S.threshold;
    return S;
  };
  let loadSeq = 0;
  const open = async (i, opts = {}) => {
    if (!S || !S.mics.length) return;
    await save();
    idx = Math.max(0, Math.min(S.mics.length - 1, i));
    const row = S.mics[idx], seq = ++loadSeq;
    ls(key, row.stem);
    Q('#pk-nav').value = idx + 1;
    Q('#pk-stem').textContent = `${idx + 1} of ${S.mics.length} · ${row.stem}`;
    let m;
    try { m = await api(`${BASE}/mic?job=${enc(job)}&stem=${enc(row.stem)}`); } catch (e) { return err(e.message); }
    if (seq !== loadSeq) return;
    M = m; undo = [];
    if (M.state === 'suggested' && (!M.suggest || M.suggest.version !== S.version) && S.version && editable() && !opts.noRefresh) {
      busy('Picking this micrograph with the new templates…');      // suggestions of older templates: again, now
      try { await api(BASE + '/suggest', {job, stems: [row.stem]}); } catch (e) { err(e.message); }
      busy('');
      if (seq !== loadSeq) return;
      return open(idx, {noRefresh: true});
    }
    const im = new Image();
    im.onload = () => { if (seq !== loadSeq) return; img = im; view = null; process(); fit(); draw(); };
    im.src = `/api/project/micraw.png?job=${enc(job)}&stem=${enc(row.stem)}`;
    const nxt = S.mics[idx + 1];                            // the next one's picture, ahead
    if (nxt) new Image().src = `/api/project/micraw.png?job=${enc(job)}&stem=${enc(nxt.stem)}`;
    if (img) draw();
    renderInfo(); renderList(); renderSide();
  };
  const refreshCounts = () => {
    if (!S || !M) return;
    const row = S.mics[idx];
    if (row && row.stem === M.stem) { row.picks = M.picks.length; row.rejected = M.rejected.length; row.state = M.state; }
    renderInfo(); renderKpi(); renderList(); renderNext();
  };

  // ------------------------------------------------------------------ actions
  const train = async () => {
    if (!editable()) return;
    await save(); err('');
    busy('Training templates from your picks…');
    try {
      const info = await api(BASE + '/train', {job});
      th = info.threshold ?? th;
      busy('Picking the next micrographs with the templates…');
      const d = await api(BASE + '/suggest', {job, after: S.mics[idx].stem});
      await loadState();
      if (d.stems.length) { busy(''); return open(S.mics.findIndex(r => r.stem === d.stems[0])); }
    } catch (e) { err(e.message); }
    busy(''); await loadState(); render();
  };
  const suggestMore = async (stems) => {
    if (!editable()) return;
    await save(); err('');
    if (!S.version) return err('Train first: the templates come from your picks.');
    busy(stems ? 'Picking this micrograph with the templates…' : 'Picking the next micrographs with the templates…');
    try {
      if (stems && M && M.state !== 'suggested') {                 // a reviewed micrograph: its picks stay, suggestions added
        M.state = 'suggested'; dirty = true; await save();
      }
      const d = await api(BASE + '/suggest', stems ? {job, stems} : {job, after: S.mics[idx].stem});
      await loadState();
      busy('');
      if (!d.stems.length) { err('Every micrograph is reviewed or has suggestions already.'); return render(); }
      return open(S.mics.findIndex(r => r.stem === d.stems[0]), {noRefresh: true});
    } catch (e) { err(e.message); }
    busy('');
  };
  const accept = async () => {
    if (!M || !editable()) return;
    if (M.state === 'suggested') {
      snapshot();
      M.picks = M.picks.concat(visibleSuggestions().map(q => [q[0], q[1], 1]));
      M.state = 'reviewed'; dirty = true; await save(); refreshCounts();
    }
    const n = S.mics.length;
    for (let k = 1; k <= n; k++) {                          // the next micrograph with suggestions to check
      const j = (idx + k) % n;
      if (S.mics[j].state === 'suggested') return open(j);
    }
    if (idx + 1 < n) open(idx + 1);
  };
  const setThreshold = async (v, persist) => {
    th = +v; Q('#pk-thv').textContent = th.toFixed(3); renderHist(); draw(); renderInfo();
    if (persist && editable()) { try { await api(BASE + '/threshold', {job, threshold: th}); S.threshold = th; } catch (e) { err(e.message); } }
  };
  const runAll = async () => {
    await save(); err('');
    const rev = S.mics.filter(r => r.state === 'reviewed').length;
    const msg = S.version ? `Pick all ${S.mics.length} micrographs: the ${rev} reviewed ones keep your picks, the others are picked with the templates at threshold ${(th ?? 0).toFixed(3)}. It runs through the queue.`
      : `No templates: only your ${rev} reviewed micrographs get picks. Finish with your picks only?`;
    if (!confirm(msg)) return;
    if (S.version && th !== S.threshold) await setThreshold(th, true);
    try { await api(BASE + '/run', {job}); } catch (e) { return err(e.message); }
    await loadState(); render(); poll();
  };
  const poll = () => {
    clearTimeout(pollTimer);
    if (!alive || !S || !['queued', 'running'].includes(S.status)) return;
    pollTimer = setTimeout(async () => {
      if (!check()) return;
      const was = S.status;
      try { await loadState(); } catch (e) {}
      renderSide(); renderNext();
      if (was !== S.status && !['queued', 'running'].includes(S.status)) open(idx);
      poll();
    }, 3000);
  };

  // ------------------------------------------------------------------ panels
  const step = () => {
    if (!S) return 0;
    if (['queued', 'running', 'finished'].includes(S.status)) return 4;
    const picks = S.mics.reduce((a, r) => a + (r.state === 'reviewed' ? r.picks : 0), 0);
    return S.version ? 3 : picks >= 20 ? 2 : 1;
  };
  const renderSteps = () => {
    const k = step();
    Q('#pk-steps').innerHTML = ['Pick by hand', 'Train', 'Check suggestions', 'Pick all'].map((t, i) =>
      `<span class="pk-step ${i + 1 < k ? 'done' : i + 1 === k ? 'now' : ''}"><b>${i + 1 < k ? '✓' : i + 1}</b>${t}</span>`).join('');
  };
  const renderNext = () => {
    renderSteps();
    if (!S) return;
    const rev = S.mics.filter(r => r.state === 'reviewed'), picks = rev.reduce((a, r) => a + r.picks, 0);
    const todo = S.mics.filter(r => r.state === 'suggested').length;
    const changed = S.train && rev.some(r => r.picks) && picks !== S.train.picks;
    let t;
    if (S.status === 'queued') t = '<b>Pick all is queued</b>: it starts when the GPU is free. The picks can change again when it has finished.';
    else if (S.status === 'running') t = '<b>Picking all micrographs…</b> Progress on the right.';
    else if (S.status === 'finished' && !S.edited_after_run) t = `<b>Done.</b> Green circles: what Pick all found. Make an <b>Extract</b> job from this job (New job → Extract) to cut the particles, or keep correcting and Pick all again.`;
    else if (!picks) t = `<b>Click on particles to pick them</b> — every particle you are sure of on this micrograph. Pick <b>20–50</b> on 2–3 micrographs, then press <b>Train &amp; suggest</b>.`;
    else if (!S.version) t = picks < 20 ? `${picks} picks on ${rev.length} micrograph${rev.length > 1 ? 's' : ''}. Pick a few more (<b>~20–50</b> in all), then <b>Train &amp; suggest (T)</b>.`
      : `${picks} picks on ${rev.length} micrographs: press <b>Train &amp; suggest (T)</b> — templates from your picks, then the next ${S.settings.suggest} micrographs picked with them.`;
    else if (M && M.state === 'suggested') t = `<b>Check the blue suggestions</b>: click a wrong one to remove it, click a missed particle to add it, or move the <b>threshold</b> if many are wrong or missing. <b>Enter</b> keeps them and opens the next one${todo > 1 ? ` (${todo} to check)` : ''}.`;
    else if (todo) t = `${todo} micrograph${todo > 1 ? 's have' : ' has'} suggestions to check: press <b>Enter</b> to go there.`;
    else if (changed) t = `Your corrections are not in the templates yet: <b>Train &amp; suggest (T)</b> again — or, when the suggestions needed few corrections, <b>Pick all micrographs</b>.`;
    else t = `All suggestions checked. <b>Suggest more (S)</b> to check a few more, or <b>Pick all micrographs</b> when they need few corrections.`;
    Q('#pk-next').innerHTML = `<span>${t}</span>`;            // fixed height: the micrograph must not move under the pointer
    const bn = Q('#pk-fin .pk-banner'); if (bn) bn.style.display = S.status === 'finished' && S.edited_after_run ? '' : 'none';
  };
  const renderKpi = () => {
    if (!S) return;
    const rev = S.mics.filter(r => r.state === 'reviewed');
    const picks = rev.reduce((a, r) => a + r.picks, 0), todo = S.mics.filter(r => r.state === 'suggested').length;
    Q('#pk-kpi').innerHTML = [[picks, 'your picks'], [`${rev.length}<small style="font-size:12px;color:var(--ink3)"> / ${S.mics.length}</small>`, 'reviewed'], [todo, 'to check']]
      .map(([v, c]) => `<div><div class="v">${typeof v === 'number' ? v.toLocaleString() : v}</div><div class="c">${c}</div></div>`).join('');
    Q('#pk-diam').textContent = `⌀ ${S.settings.diameter_a} Å`;
  };
  const renderInfo = () => {
    if (!M || !S) return;
    const sug = visibleSuggestions().length;
    const st = M.state === 'reviewed' ? 'reviewed' : M.state === 'suggested' ? 'suggestions to check' : M.output ? 'picked by Pick all' : 'not reviewed';
    Q('#pk-info').innerHTML = `${esc(st)} · <span style="color:var(--you)">${M.picks.length} yours</span>` + (M.state === 'suggested' ? ` · <span style="color:var(--data)">${sug} suggested</span>` : '')
      + (M.output && M.state !== 'reviewed' && M.state !== 'suggested' ? ` · <span style="color:var(--good)">${M.output.length} from Pick all</span>` : '')
      + (M.rejected.length ? ` · <span style="color:var(--bad)">${M.rejected.length} removed</span>` : '') + (M.defocus_a ? ` · defocus ${(M.defocus_a / 10000).toFixed(2)} µm` : '');
    Q('#pk-accept').disabled = !editable();
    Q('#pk-accept').textContent = M.state === 'suggested' ? 'Accept & next ⏎' : 'Next to check ⏎';
  };
  const chip = r => r.state === 'reviewed' ? '<span class="pk-chip reviewed">✓ done</span>' : r.state === 'suggested' ? `<span class="pk-chip ${r.stale ? 'stale' : 'suggested'}">${r.stale ? 'old' : 'check'}</span>`
    : r.output != null ? '<span class="pk-chip output">auto</span>' : '<span class="pk-chip">–</span>';
  const renderList = () => {
    if (!S) return;
    const ok = r => filter === 'all' || (filter === 'check' && r.state === 'suggested') || (filter === 'reviewed' && r.state === 'reviewed') || (filter === 'new' && !r.state);
    const rows = S.mics.map((r, i) => [r, i]).filter(([r]) => ok(r));
    Q('#pk-lcount').textContent = `${rows.length}`;
    Q('#pk-list').innerHTML = rows.map(([r, i]) => `<div class="pk-row ${i === idx ? 'on' : ''}" data-i="${i}" title="${esc(r.stem)}"><span class="n">${i + 1}</span><span class="st">${esc(r.stem)}</span>${chip(r)}<span class="n">${r.state === 'reviewed' ? r.picks : r.output ?? ''}</span></div>`).join('')
      || '<div class="pk-note">None.</div>';
    Q('#pk-list').querySelectorAll('.pk-row').forEach(e => e.onclick = () => open(+e.dataset.i));
    const on = Q('#pk-list .pk-row.on'), L = Q('#pk-list');      // within the list only: the page must not move under the pointer
    if (on && (on.offsetTop < L.scrollTop || on.offsetTop + on.offsetHeight > L.scrollTop + L.clientHeight)) L.scrollTop = on.offsetTop - L.clientHeight / 2;
    Q('#pk-filt').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.f === filter));
  };
  let thDragging = false;
  const renderSide = () => {
    if (!S) return;
    renderKpi(); renderNext();
    const ed = editable(), tr = S.train;
    Q('#pk-train').disabled = !ed || !!S.busy; Q('#pk-more').disabled = !ed || !S.version; Q('#pk-this').disabled = !ed || !S.version;
    Q('#pk-train').textContent = S.version ? 'Train again & suggest (T)' : 'Train & suggest (T)';
    Q('#pk-tm').innerHTML = tr ? `<div class="pk-note">${tr.n} template${tr.n > 1 ? 's' : ''} from ${tr.picks} picks on ${tr.micrographs} micrograph${tr.micrographs > 1 ? 's' : ''} (${tr.seconds} s)${tr.missing ? `; ${tr.missing} of your picks look unlike any template` : ''}.</div>
      <div class="pk-tm"><img src="${BASE}/templates.png?job=${enc(job)}&v=${S.version}" alt="templates" style="height:64px;width:auto;max-width:100%"></div>
      <div class="pk-note" style="margin-top:2px">picks per template: ${tr.counts.join(' · ')}</div>`
      : '<div class="pk-note">Templates appear here after training: averages of your picks, centred and rotationally aligned (2D classes).</div>';
    Q('#pk-thr').style.display = S.version ? '' : 'none';
    if (S.version) {
      const vals = [...(tr ? tr.pos : []), ...(tr ? tr.neg : [])];
      const lo = Math.max(0, Math.floor((Math.min(...vals, th ?? 1) - 0.08) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...vals, th ?? 0) + 0.05) * 20) / 20);
      const s = Q('#pk-ths'); s.min = lo; s.max = Math.max(hi, lo + 0.1); s.disabled = !ed;
      if (th !== null && !thDragging) s.value = th;
      Q('#pk-thv').textContent = th === null ? '–' : (+th).toFixed(3);
      renderHist();
    }
    // finish
    const rev = S.mics.filter(r => r.state === 'reviewed').length, p = S.progress || {};
    let f;
    if (S.status === 'queued') f = '<div class="pk-note">Queued: it runs when the GPU is free.</div>';
    else if (S.status === 'running') f = `<div class="pk-note">Picking: ${p.movies_done ?? 0} of ${p.movies_to_do ?? S.mics.length} micrographs…</div><div class="pk-prog"><div style="width:${100 * (p.movies_done || 0) / (p.movies_to_do || S.mics.length)}%"></div></div>`;
    else {
      const sm = S.summary || {};
      f = (S.status === 'finished' && sm.picks != null ? `<div class="pk-note" style="margin:0 0 8px">Last run: <b style="color:var(--ink)">${(+sm.picks).toLocaleString()}</b> picks on ${sm.micrographs} micrographs (${sm.reviewed} reviewed, ${(+sm.manual_picks).toLocaleString()} yours) in ${sm.seconds} s.</div>` : '')
        + `<button id="pk-run" class="${S.version ? 'go' : ''}" style="width:100%" ${ed && (S.version || rev) ? '' : 'disabled'}>${S.status === 'finished' ? 'Pick all again' : S.version ? 'Pick all micrographs' : 'Finish with my picks only'}</button>
          <div class="pk-note">${S.version ? `Every micrograph: the ${rev} reviewed keep your picks exactly, the other ${S.mics.length - rev} are picked with the templates at the threshold. Runs through the queue; the output goes to Extract like a Pick job's.` : 'Without templates only the micrographs you picked get picks. Train first to pick the rest.'}</div>`;
    }
    if (S.error && S.status !== 'finished') f += `<div class="pk-err">${esc(S.error)}</div>`;
    f = `<div class="pk-banner" style="display:${S.status === 'finished' && S.edited_after_run ? '' : 'none'}">You changed picks after the last Pick all: run it again to update the job's output.</div>` + f;
    Q('#pk-fin').innerHTML = f;
    const rb = Q('#pk-run'); if (rb) rb.onclick = runAll;
  };
  const renderHist = () => {
    const c = Q('#pk-hist'), tr = S && S.train;
    if (!c || !tr) return;
    const x = c.getContext('2d'), W = c.width, H = c.height, s = Q('#pk-ths'), lo = +s.min, hi = +s.max, nb = 48;
    x.clearRect(0, 0, W, H);
    const hist = vs => { const h = new Array(nb).fill(0); vs.forEach(v => { const i = Math.floor((v - lo) / (hi - lo) * nb); if (i >= 0 && i < nb) h[i]++; }); return h; };
    const sg = M && M.state === 'suggested' && M.suggest ? M.suggest.score.filter((_, i) => M.suggest.ok[i]) : [];
    const sets = [[hist(sg), css('--ink3'), 0.5], [hist(tr.pos), css('--you'), 0.9], [hist(tr.neg), css('--bad'), 0.9]];
    sets.forEach(([h, col, al]) => {
      const mx = Math.max(1, ...h); x.globalAlpha = al; x.fillStyle = col;
      h.forEach((v, i) => { const bh = (H - 14) * v / mx; x.fillRect(i * W / nb + 1, H - 12 - bh, W / nb - 2, bh); });
    });
    x.globalAlpha = 1;
    if (th !== null) { const tx = (th - lo) / (hi - lo) * W; x.fillStyle = '#fff'; x.fillRect(tx - 1.5, 0, 3, H - 10); }
    x.fillStyle = css('--ink3'); x.font = '20px sans-serif'; x.textBaseline = 'bottom';
    x.fillText(lo.toFixed(2), 2, H); const t = hi.toFixed(2); x.fillText(t, W - x.measureText(t).width - 2, H);
    const pos = tr.pos.length ? tr.pos.filter(v => v >= th).length / (tr.pos.length + tr.missing) : null;
    const neg = tr.neg.filter(v => v >= th).length;
    Q('#pk-thn').innerHTML = `<span style="color:var(--you)">■</span> your picks (scored by templates from the other half) · <span style="color:var(--bad)">■</span> removed · <span style="color:var(--ink3)">■</span> candidates here<br>`
      + (pos !== null ? `This threshold keeps <b style="color:var(--ink)">${Math.round(100 * pos)}%</b> of your picks` : '') + (tr.neg.length ? ` and ${neg} of the ${tr.neg.length} you removed.` : '.')
      + (M && M.state === 'suggested' ? ` Here: ${visibleSuggestions().length} suggestions.` : '');
  };
  const render = () => { renderSide(); renderList(); renderInfo(); draw(); };

  // ------------------------------------------------------------------ controls
  Q('#pk-prev').onclick = () => open(idx - 1);
  Q('#pk-nextm').onclick = () => open(idx + 1);
  Q('#pk-nav').oninput = e => { Q('#pk-stem').textContent = `${e.target.value} of ${S.mics.length} · ${S.mics[e.target.value - 1].stem}`; };
  Q('#pk-nav').onchange = e => open(+e.target.value - 1);
  Q('#pk-accept').onclick = accept;
  Q('#pk-train').onclick = train;
  Q('#pk-more').onclick = () => suggestMore();
  Q('#pk-this').onclick = () => suggestMore([S.mics[idx].stem]);
  Q('#pk-show').onchange = e => { show = e.target.checked; draw(); };
  const lpText = () => { Q('#pk-lp').value = lp; Q('#pk-lpv').textContent = lp ? lp + ' Å' : 'off'; };
  lpText();
  Q('#pk-lp').oninput = e => { lp = +e.target.value; lpText(); ls('celeris-lowpass', lp); process(); draw(); };
  const ths = Q('#pk-ths');
  ths.oninput = e => { thDragging = true; setThreshold(e.target.value, false); };
  ths.onchange = e => { thDragging = false; setThreshold(e.target.value, true); };
  Q('#pk-filt').querySelectorAll('button').forEach(b => b.onclick = () => { filter = b.dataset.f; renderList(); });
  const undoOne = () => {
    if (!undo.length || !editable()) return;
    const u = JSON.parse(undo.pop()); M.picks = u.picks; M.rejected = u.rejected; M.state = u.state; edited();
  };
  const onKey = ev => {
    if (!check()) return;
    const t = ev.target;
    if (t && (t.tagName === 'INPUT' && t.type !== 'range' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
    if (ev.altKey || ev.metaKey || (ev.ctrlKey && ev.key.toLowerCase() !== 'z')) return;
    if (!el.offsetParent) return;                           // not shown
    const k = ev.key;
    const step_ = () => +(ths.step) * 5;
    const map = {
      ArrowLeft: () => open(idx - 1), ArrowRight: () => open(idx + 1), Enter: accept,
      t: train, s: () => suggestMore(), p: () => suggestMore([S.mics[idx].stem]),
      h: () => { show = !show; Q('#pk-show').checked = show; draw(); },
      '[': () => S.version && setThreshold(Math.max(+ths.min, th - step_()), true), ']': () => S.version && setThreshold(Math.min(+ths.max, th + step_()), true),
      u: undoOne, z: () => ev.ctrlKey && undoOne(), '0': () => { if (view) { view.z = 1; view.ox = view.oy = 0; draw(); } },
      '+': () => { if (view) { view.z = Math.min(8, view.z * 1.25); clampView(); draw(); } }, '-': () => { if (view) { view.z = Math.max(1, view.z / 1.25); clampView(); draw(); } },
    };
    const fn = map[k] || map[k.toLowerCase()];
    if (!fn || (t && t.tagName === 'INPUT' && t.type === 'range' && k.startsWith('Arrow'))) return;
    ev.preventDefault(); fn();
  };
  document.addEventListener('keydown', onKey);
  window.addEventListener('beforeunload', flushBeacon);
  const cleanup = () => {
    if (!alive) return;
    alive = false; flushBeacon();
    clearTimeout(saveTimer); clearTimeout(pollTimer); clearInterval(watch);
    document.removeEventListener('keydown', onKey); window.removeEventListener('resize', onResize); window.removeEventListener('beforeunload', flushBeacon);
  };
  const check = () => { if (alive && !el.isConnected) cleanup(); return alive; };
  const watch = setInterval(check, 1000);

  (async () => {
    try { await loadState(); } catch (e) { el.innerHTML = `<div class="pk-err" style="padding:16px">${esc(e.message)}</div>`; return; }
    if (!S.mics.length) { Q('#pk-next').textContent = 'The MotionCtf job has no micrographs yet.'; return; }
    Q('#pk-nav').max = S.mics.length;
    const last = ls(key), i = S.mics.findIndex(r => r.stem === last);
    const first = S.mics.findIndex(r => r.state === 'suggested');
    render();
    await open(i >= 0 ? i : first >= 0 ? first : 0);
    poll();
  })();
  return {dispose: cleanup};
}
window.CelerisPicker = {mount};
})();
