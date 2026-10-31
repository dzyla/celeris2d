// job pages of their own (dash.js renderStage): Import (the session's movies) and Extract (the particles)
'use strict';
const kpiHtml = k => k.map(([v, c]) => `<div class="num"><div class="v">${v}</div><div class="c">${esc(c)}</div></div>`).join('');
const when = t => t ? new Date(t * 1000).toLocaleString(undefined, {year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) : '–';
const bytes = b => b >= 1e12 ? (b / 1e12).toFixed(2) + ' TB' : b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : (b / 1e6).toFixed(0) + ' MB';

// Import: how many movies, what they are, when they were recorded
async function stageImport(st, j) {
  st.className = 'stage';
  st.innerHTML = '<div class="launch">reading the movies…</div>';
  let d;
  try { d = await api('/api/project/jobview?job=' + encodeURIComponent(j.name)); } catch (e) { st.innerHTML = `<div class="launch err">${esc(e.message)}</div>`; return; }
  if (current !== j.name) return;
  const h = d.header || {};
  const fmts = Object.entries(d.formats || {}).map(([k, n]) => `${k.replace('.', '').toUpperCase()}${Object.keys(d.formats).length > 1 ? ' ' + n : ''}`).join(', ');
  const k = [[d.movies.toLocaleString(), 'movies'], [bytes(d.bytes), fmts + ' files']];
  if (h.frames) k.push([h.frames, h.frames > 1 ? 'frames per movie' : 'frame per file (micrographs)']);
  if (h.nx) k.push([`${h.nx} × ${h.ny}`, 'pixels per frame']);
  if (h.apix) k.push([h.apix + ' Å', 'pixel size in the header']);
  if (d.hours) k.push([d.hours >= 1 ? d.hours.toFixed(1) + ' h' : Math.round(d.hours * 60) + ' min', 'from first to last file']);
  if (d.per_hour) k.push([Math.round(d.per_hour), 'movies per hour']);
  st.innerHTML = `<div class="jv">
      <div class="mpkpi">${kpiHtml(k)}</div>
      <div class="jvgrid">
        <div class="card"><div id="imtl" class="plotbox"></div></div>
        <div class="card pad"><table class="mpt">
          <tr><td>Folder</td><td class="wrapt">${esc(d.root)}</td></tr>
          <tr><td>First movie</td><td>${esc(d.first || '–')}</td></tr>
          ${h.dtype ? `<tr><td>Data type</td><td>${esc(h.dtype)}${h.compression && h.compression !== 'NONE' ? ', ' + esc(h.compression) : ''}</td></tr>` : ''}
          ${h.mode != null ? `<tr><td>MRC mode</td><td>${h.mode}</td></tr>` : ''}
          <tr><td>First file written</td><td>${when(d.recorded_from)}</td></tr>
          <tr><td>Last file written</td><td>${when(d.recorded_to)}</td></tr>
          ${d.median_interval_s ? `<tr><td>Typical interval</td><td>${d.median_interval_s.toFixed(0)} s per movie</td></tr>` : ''}
          <tr><td>Imported</td><td>${when(d.created)}</td></tr>
          ${d.missing ? `<tr><td>Missing now</td><td class="err">${d.missing} files no longer found</td></tr>` : ''}
        </table>
        ${(d.pauses || []).length ? `<h4>Pauses (longer than 10 intervals)</h4><table class="mpt">${d.pauses.slice(0, 12).map(p => `<tr><td>after movie ${p.after}</td><td>${p.minutes} min</td></tr>`).join('')}</table>` : ''}
        ${(d.folders || []).length > 1 ? `<h4>Sub-folders</h4><table class="mpt">${d.folders.map(f => `<tr><td>${esc(f.name)}</td><td>${f.n}</td></tr>`).join('')}</table>` : ''}
        ${(d.skipped || []).length ? `<h4>Left out (truncated or unreadable)</h4><div class="muted" style="font-size:12.5px">${d.skipped.map(esc).join(', ')}</div>` : ''}
        </div></div></div>`;
  if (!d.times || d.times.length < 2) return;
  const P = await CelerisPlots.plotly(), T = THEME();
  if (!$('#imtl')) return;
  const x = d.times.map(t => new Date(t * 1000)), y = d.times.map((_, i) => i * d.times_step + 1);
  P.react($('#imtl'), [{type: 'scatter', mode: 'lines', x, y, line: {color: css('--data'), width: 2, shape: 'hv'}, hovertemplate: '%{x}<br>movie %{y}<extra></extra>'}],
    plotLayout(T, '<b>Movies over time</b> · file times (when written: recorded, or copied)', {height: 320, showlegend: false,
      xaxis: {type: 'date', title: {text: ''}}, yaxis: {title: {text: 'movies'}}}), {displaylogo: false, responsive: true});
}

// Extract: what was boxed - the particles per micrograph, the box, a random handful as images
async function stageExtract(st, j) {
  st.className = 'stage';
  const name = j.name;
  st.innerHTML = `<div class="jv">
      <div class="lhead"><span id="exstate"></span></div>
      <div class="mpkpi" id="exkpi"></div>
      <div class="card"><div class="gal-tools"><div class="sum" id="exgs">A random handful of the particles</div>
        <div class="lp" style="min-width:200px"><label for="exlp">Low-pass filter</label><input type="range" id="exlp" min="0" max="30" step="1"><span id="exlpv"></span></div>
        <button class="quiet" id="exmore">Show other particles</button></div>
        <div class="ptcl" id="exgal"><div class="muted" style="padding:20px">loading particles…</div></div></div>
      <div class="mpplots"><div id="exper" class="w2"></div><div id="exhist"></div><div id="exdef"></div></div>
    </div>`;
  let seed = 0, lp = 0, d = null, timer = null;
  lp = 10; try { lp = +(localStorage.getItem('celeris-ptcl-lowpass') ?? 10); } catch (e) {}
  const lpText = () => { $('#exlp').value = lp; $('#exlpv').textContent = lp ? lp + ' Å' : 'off'; };
  const gallery = async () => {
    let g;
    try { g = await api(`/api/project/ptcls?job=${encodeURIComponent(name)}&n=48&seed=${seed}&lowpass=${lp}`); } catch (e) { $('#exgal').innerHTML = `<div class="err" style="padding:20px">${esc(e.message)}</div>`; return; }
    if (current !== name || !$('#exgal')) return;
    if (!g.tiles.length) { $('#exgal').innerHTML = '<div class="muted" style="padding:20px">No particles yet.</div>'; return; }
    $('#exgs').innerHTML = `${g.tiles.length} of <b>${g.total.toLocaleString()}</b> particles, at random · box ${g.box} px, ${g.apix.toFixed(2)} Å/px (${(g.box * g.apix).toFixed(0)} Å)`;
    $('#exgal').innerHTML = g.tiles.map(t => `<figure title="${esc(t.stem)}${t.score != null ? ' · score ' + t.score.toFixed(3) : ''} · defocus ${(t.defocus_a / 1e4).toFixed(2)} µm"><img src="${t.src}" alt=""></figure>`).join('');
  };
  const plots = async () => {
    const P = await CelerisPlots.plotly(), T = THEME(), cfg = {displaylogo: false, responsive: true};
    if (!$('#exper')) return;
    const r = d.rows, n = r.map(x => x.particles);
    P.react($('#exper'), [{type: r.length > 2000 ? 'scattergl' : 'scatter', mode: 'markers', x: r.map(x => x.i), y: n, text: r.map(x => x.stem),
        marker: {color: css('--data'), size: 5, opacity: 0.8}, hovertemplate: '%{text}: %{y} particles<extra></extra>'}],
      plotLayout(T, '<b>Particles per micrograph</b> · in the order extracted', {showlegend: false, xaxis: {title: {text: 'micrograph'}}, yaxis: {title: {text: 'particles'}}}), cfg);
    P.react($('#exhist'), [{type: 'histogram', x: n, marker: {color: css('--data')}, nbinsx: 40}],
      plotLayout(T, '<b>Distribution</b> · particles per micrograph', {showlegend: false, xaxis: {title: {text: 'particles'}}, yaxis: {title: {text: 'micrographs'}}}), cfg);
    if (d.defocus_a) P.react($('#exdef'), [{type: 'histogram', x: d.defocus_a.map(v => v / 1e4), marker: {color: css('--data')}, nbinsx: 50}],
      plotLayout(T, '<b>Defocus</b> of the particles (sampled)', {showlegend: false, xaxis: {title: {text: 'defocus (µm)'}}, yaxis: {title: {text: 'particles'}}}), cfg);
  };
  const tick = async () => {
    clearTimeout(timer);
    if (current !== name || !$('#exkpi')) return;
    let jj;
    try { d = await api('/api/project/jobview?job=' + encodeURIComponent(name)); jj = jobs.find(x => x.name === name) || j; } catch (e) { return; }
    if (current !== name || !$('#exkpi')) return;
    const running = jj.status === 'running', mics = d.rows.length;
    const med = mics ? d.rows.map(r => r.particles).sort((a, b) => a - b)[Math.floor(mics / 2)] : 0;
    $('#exstate').innerHTML = running ? '<span class="dot running pulse"></span><b>Extracting</b> <span class="muted">· updates every 10 s</span>' : '';
    $('#exkpi').innerHTML = kpiHtml([[d.particles.toLocaleString(), 'particles'], [mics.toLocaleString(), 'micrographs'], [med, 'median per micrograph'],
      [`${d.box} px`, `box, ${d.box_a} Å`], [`${(+d.apix).toFixed(2)} Å`, 'pixel size'], ...(d.particle_a ? [[`${d.particle_a} Å`, 'particle diameter']] : []),
      ...(d.recentred ? [['yes', 're-centred on a map']] : []), [d.dtype === 'float16' ? '16 bit' : '32 bit', 'stored as']]);
    plots();
    if (!$('#exgal').querySelector('figure') || running && (Date.now() / 1000) % 60 < 10) gallery();
    if (running) timer = setTimeout(tick, 10000);
  };
  lpText();
  $('#exlp').oninput = e => { lp = +e.target.value; lpText(); };
  $('#exlp').onchange = () => { try { localStorage.setItem('celeris-ptcl-lowpass', lp); } catch (e) {} gallery(); };
  $('#exmore').onclick = () => { seed++; gallery(); };
  tick();
}
