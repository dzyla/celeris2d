// Ab initio 3D add-on for the Celeris2D dashboard.
// Self-contained: it only reads the page's status (event "celeris:status"), hears when a 2D result or the live run
// is opened ("celeris:result"), and uses the page's helpers (api, esc, toast, fmtN, fmtDur, store) and globals
// (current, sel, lastStatus, flushSave). Changing the 2D page does not need changes here, and vice versa.
"use strict";
(() => {
  const NGL_URL = "https://unpkg.com/ngl@2.4.0/dist/ngl.js";
  let defaults = null, shown = null, stage = null, comp = null, lastMaps = "", runningFor = null;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- DOM (built once, appended next to the 2D view)
  const css = document.createElement("style");
  css.textContent = `
    #maps3d { display: none; } #maps3d.has { display: flex; }
    #maps3d .chip.map { border-color: #6a4fc9; }
    #maps3d .chip.map.on { background: #2b2350; border-color: #8f74f0; color: var(--ink); }
    #ab3d { display: none; padding: 10px 14px; overflow-y: auto; }
    main.show3d #ab3d { display: block; }
    #ab3d .head { font-size: 12.5px; color: var(--ink2); margin-bottom: 8px; }
    #ab3d .head b { color: var(--ink); }
    #ab3d .grid { display: grid; grid-template-columns: minmax(320px, 1.2fr) minmax(280px, 1fr); gap: 12px; }
    #ab3dView { width: 100%; height: 460px; background: #000; border: 1px solid var(--line); border-radius: 6px; position: relative; }
    #ab3d .ctl { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 6px 0; font-size: 12px; color: var(--ink2); }
    #ab3d .ctl input[type=range] { width: 160px; }
    #ab3d figure { margin: 0 0 10px; }
    #ab3d figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
    #ab3d figcaption { font-size: 11.5px; color: var(--ink3); margin: 2px 0 6px; }
    #ab3dCfg { display: none; gap: 8px 14px; flex-wrap: wrap; padding: 6px 0 2px; }
    #ab3dCfg.show { display: flex; }
    #ab3dCfg label { display: flex; flex-direction: column; font-size: 11px; color: var(--ink3); gap: 2px; }
    #ab3dCfg input { width: 86px; }
    @media (max-width: 900px) { #ab3d .grid { grid-template-columns: 1fr; } }
    #ab3d .tiles { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 10px; }
    #ab3d .stat { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 7px 11px; min-width: 120px; }
    #ab3d .stat .v { font-size: 19px; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
    #ab3d .stat .l { font-size: 11px; color: var(--ink3); }
    #ab3d .stat .s { font-size: 11px; color: var(--ink2); }
    #ab3dFsc { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px 4px; margin-bottom: 10px; }
    #ab3dFsc svg { width: 100%; height: auto; display: block; }
    #ab3dFsc .tip { position: absolute; pointer-events: none; background: var(--panel3); border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; font-size: 11.5px; color: var(--ink); display: none; white-space: nowrap; }
    #ab3dFsc .key { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--ink2); margin: 2px 0 4px; }
    #ab3dFsc .key i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }`;
  document.head.appendChild(css);

  const bar = document.createElement("div");
  bar.className = "bar"; bar.id = "maps3d";
  $("results").after(bar);
  const panel = document.createElement("div");
  panel.id = "ab3d";
  $("gallery").after(panel);

  // the launcher sits in the 2D toolbar's "Next" row; settings in a collapsible row under it
  const nextRow = [...document.querySelectorAll("#toolbar .row")].find(r => /Next/.test(r.textContent));
  const launch = document.createElement("span");
  launch.className = "row"; launch.style.gap = "6px";
  launch.innerHTML = `<span class="vsep"></span>
    <button class="primary" id="ab3dRun" title="Build a 3D initial model from the selected classes (all populated classes if none are selected) and their particles">Ab initio 3D</button>
    <label class="row muted" style="gap:4px" title="Point group imposed from the start: C1, Cn, Dn">sym <input id="ab3dSym" value="C1" style="width:52px" spellcheck="false"></label>
    <button class="link" id="ab3dMore" title="Ab initio settings">3D settings</button>`;
  (nextRow || $("toolbar")).appendChild(launch);
  const cfg = document.createElement("div");
  cfg.id = "ab3dCfg";
  $("toolbar").appendChild(cfg);

  // ---------------------------------------------------------------- settings
  async function loadDefaults() {
    try { defaults = await api("/api/abinitio/defaults"); } catch (e) { return; }
    const saved = store.get("ab3d", {});
    cfg.innerHTML = Object.entries(defaults.values).filter(([k]) => k !== "sym").map(([k, v]) =>
      `<label title="${esc(defaults.help[k] || "")}">${esc(k.replace(/_/g, " "))}<input data-k="${esc(k)}" ${typeof v === "string" ? 'type="text" size="4"' : 'type="number" step="any"'} value="${esc(saved[k] ?? v)}"></label>`).join("") +
      `<label style="justify-content:flex-end"><button class="link" id="ab3dReset">reset</button></label>`;
    $("ab3dSym").value = saved.sym || defaults.values.sym;
    cfg.querySelectorAll("input").forEach(i => i.onchange = saveCfg);
    $("ab3dSym").onchange = saveCfg;
    $("ab3dReset").onclick = () => { store.set("ab3d", {}); loadDefaults(); };
  }
  function cfgValues() {
    const v = {sym: $("ab3dSym").value.trim().toUpperCase() || "C1"};
    cfg.querySelectorAll("input[data-k]").forEach(i => { if (i.value !== "") v[i.dataset.k] = i.type === "text" ? i.value.trim() : Number(i.value); });
    return v;
  }
  function saveCfg() { store.set("ab3d", cfgValues()); }
  $("ab3dMore").onclick = () => cfg.classList.toggle("show");

  // ---------------------------------------------------------------- run
  $("ab3dRun").onclick = async () => {
    if (!current || current === "__live__") { toast("open a 2D result first", true); return; }
    try {
      await flushSave();                                   // the selection must be on the server first
      const on = sel.size ? `${current}:good` : current;
      const p = cfgValues();
      const name = `${current}${sel.size ? "_good" : ""}_3d${p.sym !== "C1" ? "_" + p.sym.toLowerCase() : ""}${p.n_models > 1 ? "_k" + p.n_models : ""}`;
      const r = await api("/api/abinitio", Object.assign({on, name}, p));
      runningFor = r.name;
      toast(`ab initio ${r.name} started (${sel.size ? sel.size + " selected classes" : "all classes"}, ${p.sym})`);
      poll();
    } catch (e) { toast(e.message, true); }
  };

  // ---------------------------------------------------------------- map chips + view
  function renderChips(s) {
    const maps = s.loaded ? (s.maps || []) : [];
    const j = s.job, running = j && j.kind === "abinitio" && !j.done ? j.name : null;
    const key = JSON.stringify([maps, running, shown]);
    if (key === lastMaps) return;
    lastMaps = key;
    bar.classList.toggle("has", maps.length > 0 || !!running);
    bar.innerHTML = `<span class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">3D maps</span>` +
      maps.map(m => `<span class="chip map ${shown === m.name ? "on" : ""}" data-m="${esc(m.name)}" title="${esc(`${m.name}: from ${m.on}, ${m.classes} classes, ${fmtN(m.particles)} particles, ${m.sym}, took ${fmtDur(m.seconds || 0)}`)}">◆ ${esc(m.name)} · ${m.models > 1 ? m.models + " 3D classes" : m.resolution_a + " Å"} · ${esc(m.sym || "C1")} · ${fmtDur(m.seconds || 0)}</span>`).join("") +
      (running ? `<span class="chip live">● ${esc(running)} running</span>` : "");
    bar.querySelectorAll(".chip[data-m]").forEach(c => c.onclick = () => showMap(c.dataset.m));
  }

  function loadNGL() {
    if (window.NGL) return Promise.resolve(window.NGL);
    return new Promise((ok, bad) => {
      const sc = document.createElement("script");
      sc.src = NGL_URL; sc.onload = () => ok(window.NGL); sc.onerror = () => bad(new Error("3D viewer (NGL) could not be loaded"));
      document.head.appendChild(sc);
    });
  }

  async function showMap(name) {
    shown = name;
    document.querySelector("main").classList.add("show3d");
    document.querySelectorAll("#results .chip").forEach(c => c.classList.remove("on"));
    const m = ((lastStatus && lastStatus.maps) || []).find(x => x.name === name) || {name};
    let summ = {};
    try { summ = await api(`/api/map/${encodeURIComponent(name)}/summary.json`); } catch (e) { /* older map */ }
    const u = (f) => `/api/map/${encodeURIComponent(name)}/${f}?t=${Date.now()}`;
    const p = summ.params || {};
    panel.innerHTML = `<div class="head"><b>${esc(name)}</b> · from ${esc(summ.on || m.on || "?")} · ${summ.classes ?? "?"} classes ·
        ${esc(p.sym || "C1")}${summ.relaxed_from ? ` (solved in ${esc(summ.relaxed_from)}, refined without symmetry)` : ""} · box ${summ.box} @ ${summ.apix} Å</div>
      ${tilesHTML(summ)}
      ${modelsHTML(summ)}
      <div class="grid">
        <div>
          <div id="ab3dView"></div>
          <div class="ctl">
            <label>threshold <input type="range" id="ab3dIso" min="0.5" max="6" step="0.05" value="${store.get("ab3dIso", 2.0)}"> <span id="ab3dIsoV"></span> σ</label>
            <label><input type="checkbox" id="ab3dSpin"> spin</label>
            <label>map <select id="ab3dWhich">${mapOptions(summ)}</select></label>
            <a href="${u("map.mrc")}" download="${esc(name)}.mrc">download map.mrc</a>
          </div>
          <figure><img src="${u("map_views.png")}" alt="projections and central sections of the map"></figure>
          <div id="ab3dFsc">${fscChart(summ)}</div>
        </div>
        <div>
          <figure><img src="${u("reprojections.png")}" alt="class averages vs reprojections"><figcaption>each 2D class average next to the map's projection at the orientation given to it — they should match (cc = correlation)</figcaption></figure>
          <figure><img src="${u("views.png")}" alt="viewing directions"><figcaption>particle viewing directions on an equal-area sky map (holes = missing views, bright = crowded)</figcaption></figure>
        </div>
      </div>`;
    renderChips(lastStatus || {});
    wireFsc(summ);
    const iso = $("ab3dIso"), isoV = $("ab3dIsoV");
    isoV.textContent = Number(iso.value).toFixed(2);
    try {
      const NGL = await loadNGL();
      if (shown !== name) return;
      if (stage) { stage.dispose(); stage = null; }
      stage = new NGL.Stage("ab3dView", {backgroundColor: "black"});
      const load = async (f) => {
        stage.removeAllComponents();
        comp = await stage.loadFile(u(f), {ext: "mrc"});
        comp.addRepresentation("surface", {isolevelType: "sigma", isolevel: Number(iso.value), color: "#9fb7ff", smooth: 1});
        comp.autoView();
      };
      await load($("ab3dWhich").value);
      iso.oninput = () => { isoV.textContent = Number(iso.value).toFixed(2); store.set("ab3dIso", Number(iso.value));
        if (comp) comp.reprList.forEach(r => r.setParameters({isolevel: Number(iso.value)})); };
      $("ab3dSpin").onchange = (e) => stage.setSpin(e.target.checked);
      $("ab3dWhich").onchange = (e) => load(e.target.value);
      window.addEventListener("resize", () => stage && stage.handleResize());
    } catch (e) {
      $("ab3dView").innerHTML = `<div class="muted" style="padding:20px">${esc(e.message)} — the figures below still show the map; download map.mrc to view it in ChimeraX.</div>`;
    }
  }

  const fmtS = (x) => x == null ? "?" : x < 60 ? `${Math.round(x)} s` : `${Math.floor(x / 60)} min ${Math.round(x % 60)} s`;
  function tilesHTML(summ) {
    const t = (v, l, sub) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
    const cc = summ.reprojection_cc || [];
    const med = cc.length ? [...cc].sort((a, b) => a - b)[Math.floor(cc.length / 2)] : null;
    const tot = summ.seconds_total ?? ((summ.seconds_classes || 0) + (summ.seconds_particles || 0));
    return `<div class="tiles">` +
      t(`${summ.resolution_solvent_mask_a ?? summ.resolution_a ?? "?"} Å`, "resolution, solvent mask",
        `sphere mask ${summ.resolution_a ?? "?"} Å · no mask ${summ.resolution_unmasked_a ?? "?"} Å`) +
      t(fmtN(summ.particles_used ?? summ.particles), "particles in the map",
        summ.particles_dropped ? `${fmtN(summ.particles_dropped)} dropped (worst fit)` : `of ${fmtN(summ.particles)}`) +
      t(fmtS(tot), "time", `classes ${fmtS(summ.seconds_classes)} · SGD ${fmtS(summ.seconds_sgd)} · refine ${fmtS((summ.seconds_particles || 0) - (summ.seconds_sgd || 0))}`) +
      (med != null ? t(med.toFixed(2), "classes vs reprojections", "median correlation (1 = identical)") : "") +
      (summ.view_coverage ? t(`${Math.round(100 * summ.view_coverage.occupied)}%`, "viewing directions covered",
        summ.view_coverage.warning ? `<span style="color:var(--warn)">⚠ narrow range of views: map smeared along missing ones</span>`
          : `busiest 10% hold ${Math.round(100 * summ.view_coverage.top10_share)}%`) : "") +
      `</div>`;
  }
  // half-map FSC, drawn here so it can be inspected point by point (hover)
  const FSC_SERIES = [["fsc_solvent", "solvent mask", "#199e70"], ["fsc", "soft sphere mask", "#3987e5"], ["fsc_unmasked", "no mask", "#d95926"]];
  const FW = 520, FH = 250, FL = 40, FR = 12, FT = 10, FB = 34;
  function fscChart(summ) {
    const box = summ.box_a, nyq = 1 / (2 * (summ.apix || 1));
    const ser = FSC_SERIES.filter(([k]) => Array.isArray(summ[k]) && summ[k].length > 2);
    if (!ser.length || !box) return `<div class="muted">no FSC recorded</div>`;
    const X = (f) => FL + (f / nyq) * (FW - FL - FR), Y = (v) => FT + (1 - (v + 0.05) / 1.1) * (FH - FT - FB);
    const ticks = [40, 20, 15, 10, 8, 6, 5, 4, 3.5, 3, 2.5].filter(r => 1 / r <= nyq * 1.001);
    let g = ticks.map(r => `<line x1="${X(1 / r)}" x2="${X(1 / r)}" y1="${FT}" y2="${FH - FB}" stroke="var(--line)" stroke-width="1"/>` +
      `<text x="${X(1 / r)}" y="${FH - FB + 15}" fill="var(--ink3)" font-size="11" text-anchor="middle">${r}</text>`).join("");
    g += [0, 0.5, 1].map(v => `<text x="${FL - 6}" y="${Y(v) + 4}" fill="var(--ink3)" font-size="11" text-anchor="end">${v}</text>`).join("");
    g += `<line x1="${FL}" x2="${FW - FR}" y1="${Y(0.143)}" y2="${Y(0.143)}" stroke="var(--ink3)" stroke-dasharray="4 3"/>`;
    g += `<text x="${FW - FR}" y="${Y(0.143) - 5}" fill="var(--ink3)" font-size="10.5" text-anchor="end">0.143</text>`;
    g += `<text x="${(FL + FW - FR) / 2}" y="${FH - 3}" fill="var(--ink2)" font-size="11.5" text-anchor="middle">resolution (Å)</text>`;
    let key = "";
    for (const [k, lab, col] of ser) {
      const f = summ[k];
      const pts = f.map((v, i) => i ? `${X(i / box).toFixed(1)},${Y(v).toFixed(1)}` : null).filter(Boolean).join(" ");
      g += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;
      let ci = f.findIndex((v, i) => i > 0 && v < 0.143);
      const res = ci > 0 ? box / (ci - 1 + (f[ci - 1] - 0.143) / Math.max(f[ci - 1] - f[ci], 1e-9)) : null;
      if (res) g += `<circle cx="${X(1 / res)}" cy="${Y(0.143)}" r="4.5" fill="${col}" stroke="var(--panel)" stroke-width="2"/>`;
      key += `<span><i style="background:${col}"></i>${lab} <b style="color:var(--ink)">${res ? res.toFixed(1) + " Å" : "beyond Nyquist"}</b></span>`;
    }
    g += `<line id="ab3dFscX" y1="${FT}" y2="${FH - FB}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/>`;
    g += `<rect id="ab3dFscHit" x="${FL}" y="${FT}" width="${FW - FL - FR}" height="${FH - FT - FB}" fill="transparent"/>`;
    return `<div class="key">${key}</div><svg viewBox="0 0 ${FW} ${FH}" role="img" aria-label="half-map FSC curves">${g}</svg><div class="tip" id="ab3dFscTip"></div>`;
  }
  function wireFsc(summ) {
    const hit = $("ab3dFscHit"), tip = $("ab3dFscTip"), xl = $("ab3dFscX");
    if (!hit) return;
    const box = summ.box_a, nyq = 1 / (2 * (summ.apix || 1));
    const ser = FSC_SERIES.filter(([k]) => Array.isArray(summ[k]));
    hit.onmousemove = (e) => {
      const svg = hit.ownerSVGElement, pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const f = Math.max(1e-6, (p.x - FL) / (FW - FL - FR) * nyq), i = Math.max(1, Math.round(f * box));
      xl.setAttribute("x1", FL + (i / box) / nyq * (FW - FL - FR)); xl.setAttribute("x2", xl.getAttribute("x1")); xl.setAttribute("visibility", "visible");
      tip.innerHTML = `<b>${(box / i).toFixed(2)} Å</b>` + ser.map(([k, lab, col]) => summ[k][i] == null ? "" :
        `<div><span style="color:${col}">●</span> ${lab}: ${summ[k][i].toFixed(3)}</div>`).join("");
      const r = $("ab3dFsc").getBoundingClientRect();
      tip.style.display = "block";
      tip.style.left = Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 6) + "px";
      tip.style.top = (e.clientY - r.top + 10) + "px";
    };
    hit.onmouseleave = () => { tip.style.display = "none"; xl.setAttribute("visibility", "hidden"); };
  }

  function mapOptions(summ) {
    const ms = summ.models || [];
    if (ms.length > 1)
      return ms.map(m => `<option value="map_${m.model}.mrc">3D class ${m.model} · ${fmtN(m.particles)} · ${m.resolution_a ?? "merged"} Å</option>`).join("") +
        `<option value="consensus.mrc">consensus (all particles)</option>`;
    return `<option value="map.mrc">final (solvent mask)</option><option value="map_nomask.mrc">final, no mask</option>` +
      `<option value="from_classes.mrc">from classes</option><option value="half1_unfil.mrc">half 1 (unfiltered)</option>` +
      `<option value="half2_unfil.mrc">half 2 (unfiltered)</option><option value="mask.mrc">solvent mask</option>`;
  }
  function modelsHTML(summ) {
    const ms = summ.models || [];
    if (ms.length < 2) return "";
    const tot = ms.reduce((a, m) => a + m.particles, 0) || 1;
    return `<div class="head">3D classes: ${ms.map(m => `<b>${m.model}</b> ${fmtN(m.particles)} (${(100 * m.particles / tot).toFixed(0)}%) · ${m.resolution_a ?? "merged"} Å`).join(" &nbsp;|&nbsp; ")}</div>`;
  }

  function hideMap() {
    if (!shown) return;
    shown = null;
    document.querySelector("main").classList.remove("show3d");
    if (stage) { stage.dispose(); stage = null; }
    renderChips(lastStatus || {});
  }

  // ---------------------------------------------------------------- page events
  window.addEventListener("celeris:result", hideMap);
  window.addEventListener("celeris:status", (ev) => {
    const s = ev.detail;
    renderChips(s);
    const j = s.job;
    $("ab3dRun").disabled = !s.loaded || !!s.busy;
    if (runningFor && j && j.kind === "abinitio" && j.name === runningFor && j.done) {
      const fin = runningFor; runningFor = null;
      if (!j.error) { toast(`${fin} finished in ${fmtDur(j.seconds || 0)}`); showMap(fin); }
    } else if (!runningFor && j && j.kind === "abinitio" && !j.done) runningFor = j.name;   // started elsewhere / page reloaded
    if (shown && !(s.maps || []).some(m => m.name === shown)) hideMap();
  });
  loadDefaults();
})();
