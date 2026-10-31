// Interactive versions of the saved figures: x.plot.json (figjson.py) drawn with Plotly in the same layout, colours
// and axes as x.png, with zoom, pan and hover. window.CelerisPlots.render(div, spec, theme) -> Promise.
(function () {
  'use strict';
  let loading = null;
  function plotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (!loading) loading = new Promise((ok, bad) => {
      const s = document.createElement('script');
      s.src = '/web/plotly.js';
      s.onload = () => ok(window.Plotly);
      s.onerror = () => bad(new Error('Plotly could not be loaded'));
      document.head.appendChild(s);
    });
    return loading;
  }
  const DASH = {'--': 'dash', 'dashed': 'dash', ':': 'dot', 'dotted': 'dot', '-.': 'dashdot', 'dashdot': 'dashdot'};
  const SYMBOL = {'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down', 'D': 'diamond', 'x': 'x',
                  '+': 'cross', '*': 'star', '.': 'circle', 'd': 'diamond'};
  const px = pt => pt * 96 / 72;                       // matplotlib points -> CSS pixels

  // opts.minFont: no text smaller (px; a figure drawn small, e.g. beside the map viewer); opts.noTitles: without the
  // figure's own title lines (the card says what it is)
  function build(spec, theme, width, opts = {}) {
    const scale = width / (spec.width * 96);
    const F = v => Math.max(opts.minFont || 0, v);
    const traces = [], ann = [], images = [], shapes = [];
    const layout = {
      width, height: Math.round(spec.height * 96 * scale), margin: {l: 0, r: 0, t: 0, b: 0},
      paper_bgcolor: theme.surface, plot_bgcolor: theme.surface, showlegend: false, hovermode: 'closest',
      font: {family: theme.font, size: F(Math.max(9, 11 * scale)), color: theme.ink2},
      hoverlabel: {bgcolor: theme.well, bordercolor: theme.line, font: {color: theme.ink, family: theme.font}},
      dragmode: 'zoom',
    };
    (opts.noTitles ? [] : spec.texts || []).forEach(t => ann.push({
      text: esc(t.text).replace(/\n/g, '<br>'), xref: 'paper', yref: 'paper', x: t.x, y: t.y, xanchor: 'left',
      yanchor: 'top', showarrow: false, align: 'left',
      font: {size: F(px(t.size) * scale), color: t.color || theme.ink, family: theme.font},
    }));
    spec.panels.forEach((p, i) => {
      if (p.error || !p.pos) return;
      if (p.raster) {                                   // a picture of the panel (projections, meshes, colour bars)
        images.push({source: p.raster, xref: 'paper', yref: 'paper', x: p.pos[0], y: p.pos[3], sizex: p.pos[2] - p.pos[0],
          sizey: p.pos[3] - p.pos[1], sizing: 'stretch', layer: 'above', xanchor: 'left', yanchor: 'top'});
        return;
      }
      const n = i + 1, xa = 'xaxis' + (n > 1 ? n : ''), ya = 'yaxis' + (n > 1 ? n : '');
      const xr = 'x' + (n > 1 ? n : ''), yr = 'y' + (n > 1 ? n : '');
      const hasData = p.lines.length + p.scatter.length + p.bars.length + p.fills.length + p.segments.length + p.images.length;
      const onlyImage = p.images.length && !(p.lines.length + p.scatter.length + p.bars.length);
      const axis = (lim, scaleT, label, ticks, dom, other) => {
        const a = {domain: dom, anchor: other, range: scaleT === 'log' ? lim.map(v => Math.log10(Math.max(v, 1e-30))) : lim,
                   type: scaleT === 'log' ? 'log' : 'linear', showgrid: false, zeroline: false,
                   linecolor: theme.axis, tickcolor: theme.ink3, ticks: 'outside', ticklen: 3,
                   tickfont: {color: theme.ink2, size: F(Math.max(8, 10 * scale))},
                   title: label ? {text: esc(label), font: {color: theme.ink2, size: F(Math.max(9, 11 * scale))}, standoff: 4} : undefined,
                   showline: true, automargin: false, fixedrange: false};
        if (ticks) { a.tickvals = ticks.v; a.ticktext = ticks.text.map(esc); }
        if (!p.axis_on || onlyImage) Object.assign(a, {visible: false});
        return a;
      };
      layout[xa] = axis(p.xlim, p.xscale, p.xlabel, p.xticks, [p.pos[0], p.pos[2]], yr);
      layout[ya] = axis(p.ylim, p.yscale, p.ylabel, p.yticks, [p.pos[1], p.pos[3]], xr);
      if (p.grid_y) Object.assign(layout[ya], {showgrid: true, gridcolor: theme.grid});
      if (onlyImage) { layout[ya].scaleanchor = xr; layout[ya].constrain = 'domain'; layout[xa].constrain = 'domain'; }
      if (p.title) ann.push({text: '<b>' + esc(p.title) + '</b>', xref: 'paper', yref: 'paper', x: p.pos[0],
        y: p.pos[3], xanchor: 'left', yanchor: 'bottom', showarrow: false, yshift: 4,
        font: {size: F(Math.max(10, 13.5 * scale)), color: theme.ink}});
      if (!hasData) return;
      const common = {xaxis: xr, yaxis: yr, legendgroup: 'p' + n};
      p.fills.forEach(f => traces.push(Object.assign({type: 'scatter', mode: 'lines', x: f.x, y: f.y, fill: 'toself',
        fillcolor: rgba(f.color, f.alpha == null ? 0.3 : f.alpha), line: {width: 0}, hoverinfo: 'skip',
        name: f.label || ''}, common)));
      if (p.bars.length) {
        const byColor = {};
        p.bars.forEach(b => (byColor[b.color] = byColor[b.color] || []).push(b));
        Object.entries(byColor).forEach(([c, bs]) => traces.push(Object.assign({type: 'bar',
          x: bs.map(b => b.x + b.w / 2), y: bs.map(b => b.h), base: bs.map(b => b.y), width: bs.map(b => Math.abs(b.w)),
          marker: {color: c, line: {width: 0}}, hovertemplate: '%{x:.4g}: %{y:.4g}<extra></extra>'}, common)));
        layout.barmode = 'overlay';
      }
      p.segments.forEach(sg => traces.push(Object.assign({type: 'scattergl', mode: 'lines', x: sg.x, y: sg.y,
        line: {color: sg.color, width: px(sg.width) * scale}, opacity: sg.alpha == null ? 1 : sg.alpha,
        hoverinfo: 'skip'}, common)));
      p.lines.forEach(l => {
        const hasMarker = l.marker && l.marker !== 'None' && l.marker !== '' && l.marker !== ' ';
        const hasLine = l.dash !== 'None' && l.dash !== '' && l.width > 0;
        const mode = [hasLine ? 'lines' : '', hasMarker ? 'markers' : ''].filter(Boolean).join('+') || 'lines';
        traces.push(Object.assign({type: l.x.length > 3000 ? 'scattergl' : 'scatter', mode, x: l.x, y: l.y,
          name: l.label || '', opacity: l.alpha == null ? 1 : l.alpha,
          line: {color: l.color, width: Math.max(0.5, px(l.width) * scale), dash: DASH[l.dash] || 'solid'},
          marker: {color: l.color, size: Math.max(3, px(l.size) * scale), symbol: SYMBOL[l.marker] || 'circle'},
          hovertemplate: (l.label ? esc(l.label) + '<br>' : '') + '%{x:.4g}, %{y:.4g}<extra></extra>'}, common));
      });
      p.scatter.forEach(sc => {
        const per = sc.colors.length > 1;
        traces.push(Object.assign({type: sc.x.length > 2000 ? 'scattergl' : 'scatter', mode: 'markers', x: sc.x, y: sc.y,
          name: sc.label || '', marker: {color: per ? sc.colors : sc.colors[0], opacity: sc.alpha == null ? 1 : sc.alpha,
            size: Math.max(2, Math.sqrt(sc.sizes[0] || 9) * px(1) * scale)},
          text: sc.values, hovertemplate: (sc.label ? esc(sc.label) + '<br>' : '') + '%{x:.4g}, %{y:.4g}' +
            (sc.values ? '<br>value %{text:.4g}' : '') + '<extra></extra>'}, common));
      });
      p.images.forEach(im => {
        const [l, r, b, t] = im.extent;
        const yTop = p.ylim[1] >= p.ylim[0] ? Math.max(t, b) : Math.min(t, b);   // the value at the visual top
        images.push({source: im.src, xref: xr, yref: yr, x: Math.min(l, r), y: yTop, sizex: Math.abs(r - l), sizey: Math.abs(t - b),
          sizing: 'stretch', layer: onlyImage ? 'above' : 'below', xanchor: 'left', yanchor: 'top'});
        if (im.z) {                                       // small heatmaps: the values for hover
          const h = im.z.length, w = im.z[0].length;
          const xs = Array.from({length: w}, (_, k) => l + (k + 0.5) * (r - l) / w);
          const ys = Array.from({length: h}, (_, k) => (im.origin === 'lower' ? b + (k + 0.5) * (t - b) / h : t - (k + 0.5) * (t - b) / h));
          traces.push(Object.assign({type: 'heatmap', x: xs, y: ys, z: im.z, opacity: 0, showscale: false,
            hovertemplate: '%{x:.4g}, %{y:.4g}: %{z:.4g}<extra></extra>'}, common));
        }
      });
      if (p.legend && p.legend.items && p.legend.items.length) {     // the legend where matplotlib put it
        const L = p.legend;
        const rows = (L.title ? [`<span style="color:${theme.ink3}">${esc(L.title)}</span>`] : [])
          .concat(L.items.map(it => `<span style="color:${it.color || theme.ink2}">━━</span> ${esc(it.label)}`));
        ann.push({text: rows.join('<br>'), xref: 'paper', yref: 'paper', x: L.box[0], y: L.box[3], xanchor: 'left',
          yanchor: 'top', showarrow: false, align: 'left', font: {size: F(Math.max(8, 11 * scale)), color: theme.ink2}});
      }
      p.texts.forEach(t => {
        const fr = t.ref === 'axes';
        ann.push({text: esc(t.text).replace(/\n/g, '<br>'), showarrow: false,
          xref: fr ? 'paper' : xr, yref: fr ? 'paper' : yr,
          x: fr ? p.pos[0] + (p.pos[2] - p.pos[0]) * t.x : t.x, y: fr ? p.pos[1] + (p.pos[3] - p.pos[1]) * t.y : t.y,
          xanchor: {left: 'left', right: 'right', center: 'center'}[t.ha] || 'left',
          yanchor: {top: 'top', bottom: 'bottom', center: 'middle', baseline: 'bottom', center_baseline: 'middle'}[t.va] || 'middle',
          align: t.ha === 'right' ? 'right' : 'left',
          font: {size: F(Math.max(8, px(t.size) * scale)), color: t.color || theme.ink2}});
      });
    });
    if (!layout.xaxis) layout.xaxis = {visible: false};      // a picture-only first panel: no default axes
    if (!layout.yaxis) layout.yaxis = {visible: false};
    layout.annotations = ann;
    layout.images = images;
    layout.shapes = shapes;
    return {traces, layout};
  }
  function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c])); }
  function rgba(hex, a) {
    if (!hex) return `rgba(128,128,128,${a})`;
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  async function render(div, spec, theme, opts = {}) {
    const P = await plotly();
    const width = Math.max(opts.minWidth ?? 320, div.clientWidth || 900);
    const {traces, layout} = build(spec, theme, width, opts);
    await P.react(div, traces, layout, {displaylogo: false, responsive: false, scrollZoom: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
      toImageButtonOptions: {format: 'svg', filename: 'celeris2d_plot'}});
    return div;
  }
  window.CelerisPlots = {render, build, plotly};
})();
