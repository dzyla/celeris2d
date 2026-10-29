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
    @media (max-width: 900px) { #ab3d .grid { grid-template-columns: 1fr; } }`;
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
      `<label title="${esc(defaults.help[k] || "")}">${esc(k.replace(/_/g, " "))}<input data-k="${esc(k)}" type="number" step="any" value="${esc(saved[k] ?? v)}"></label>`).join("") +
      `<label style="justify-content:flex-end"><button class="link" id="ab3dReset">reset</button></label>`;
    $("ab3dSym").value = saved.sym || defaults.values.sym;
    cfg.querySelectorAll("input").forEach(i => i.onchange = saveCfg);
    $("ab3dSym").onchange = saveCfg;
    $("ab3dReset").onclick = () => { store.set("ab3d", {}); loadDefaults(); };
  }
  function cfgValues() {
    const v = {sym: $("ab3dSym").value.trim().toUpperCase() || "C1"};
    cfg.querySelectorAll("input[data-k]").forEach(i => { if (i.value !== "") v[i.dataset.k] = Number(i.value); });
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
      const name = `${current}${sel.size ? "_good" : ""}_3d${p.sym !== "C1" ? "_" + p.sym.toLowerCase() : ""}`;
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
      maps.map(m => `<span class="chip map ${shown === m.name ? "on" : ""}" data-m="${esc(m.name)}" title="${esc(`${m.name}: from ${m.on}, ${m.classes} classes, ${fmtN(m.particles)} particles, ${m.sym}, took ${fmtDur(m.seconds || 0)}`)}">◆ ${esc(m.name)} · ${m.resolution_a} Å · ${esc(m.sym || "C1")} · ${fmtDur(m.seconds || 0)}</span>`).join("") +
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
    panel.innerHTML = `<div class="head"><b>${esc(name)}</b> · from ${esc(summ.on || m.on || "?")} · ${summ.classes ?? "?"} classes · ${fmtN(summ.particles)} particles ·
        ${esc(p.sym || "C1")} · <b>${summ.resolution_a ?? "?"} Å</b> (half-map FSC 0.143) · classes ${summ.seconds_classes ?? "?"} s + particles ${summ.seconds_particles ?? "?"} s ·
        box ${summ.box} @ ${summ.apix} Å</div>
      <div class="grid">
        <div>
          <div id="ab3dView"></div>
          <div class="ctl">
            <label>threshold <input type="range" id="ab3dIso" min="0.5" max="6" step="0.05" value="${store.get("ab3dIso", 2.0)}"> <span id="ab3dIsoV"></span> σ</label>
            <label><input type="checkbox" id="ab3dSpin"> spin</label>
            <label>map <select id="ab3dWhich"><option value="map.mrc">final</option><option value="from_classes.mrc">from classes</option><option value="half1_unfil.mrc">half 1 (unfiltered)</option><option value="half2_unfil.mrc">half 2 (unfiltered)</option></select></label>
            <a href="${u("map.mrc")}" download="${esc(name)}.mrc">download map.mrc</a>
          </div>
          <figure><img src="${u("map_views.png")}" alt="projections of the map"><figcaption>projections of the map along z, y, x</figcaption></figure>
          <figure><img src="${u("fsc.png")}" alt="half-map FSC" style="max-width:420px"></figure>
        </div>
        <div>
          <figure><img src="${u("reprojections.png")}" alt="class averages vs reprojections"><figcaption>each class average (top) above the map's projection at the orientation assigned to it — they should match</figcaption></figure>
          <figure><img src="${u("views.png")}" alt="viewing directions" style="max-width:480px"><figcaption>particle viewing directions (holes = missing views)</figcaption></figure>
        </div>
      </div>`;
    renderChips(lastStatus || {});
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
      await load("map.mrc");
      iso.oninput = () => { isoV.textContent = Number(iso.value).toFixed(2); store.set("ab3dIso", Number(iso.value));
        if (comp) comp.reprList.forEach(r => r.setParameters({isolevel: Number(iso.value)})); };
      $("ab3dSpin").onchange = (e) => stage.setSpin(e.target.checked);
      $("ab3dWhich").onchange = (e) => load(e.target.value);
      window.addEventListener("resize", () => stage && stage.handleResize());
    } catch (e) {
      $("ab3dView").innerHTML = `<div class="muted" style="padding:20px">${esc(e.message)} — the figures below still show the map; download map.mrc to view it in ChimeraX.</div>`;
    }
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
