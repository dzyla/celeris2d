// 3D map viewer: the map as an 8-bit 3D texture (gallery.volume_payload), its isosurface found by raymarching on the
// GPU (WebGL2) - no surface mesh, so the threshold slider is instant. Lit by the density gradient; optionally coloured
// by a second volume (local resolution). Drag to rotate, wheel to zoom, double-click to reset.
// window.CelerisVolume.create(canvas) -> viewer {load(url), setLevel(v), setColor(on), view(axis), dispose()}.
(function () {
  'use strict';
  const VS = `#version 300 es
  in vec2 p; out vec2 uv; void main() { uv = p; gl_Position = vec4(p, 0.0, 1.0); }`;
  const FS = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 uv; out vec4 o;
  uniform sampler3D vol; uniform sampler3D col; uniform bool useCol;
  uniform mat3 rot; uniform float dist; uniform float iso; uniform vec3 ext; uniform float aspect;
  uniform vec3 base; uniform vec3 bg; uniform float steps;
  vec3 ramp(float t) {            // local resolution: blue (finer) -> white -> red (coarser)
    t = clamp(t, 0.0, 1.0);
    return t < 0.5 ? mix(vec3(0.16, 0.42, 0.82), vec3(0.93, 0.93, 0.92), t * 2.0)
                   : mix(vec3(0.93, 0.93, 0.92), vec3(0.85, 0.25, 0.22), t * 2.0 - 1.0);
  }
  float d(vec3 q) { return texture(vol, q).r; }
  void main() {
    vec3 ro = rot * vec3(0.0, 0.0, dist);
    vec3 rd = normalize(rot * vec3(uv.x * aspect, uv.y, -1.8));
    vec3 lo = -ext * 0.5, hi = ext * 0.5;
    vec3 inv = 1.0 / rd;
    vec3 t0 = (lo - ro) * inv, t1 = (hi - ro) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    float tn = max(max(tmin.x, tmin.y), tmin.z), tf = min(min(tmax.x, tmax.y), tmax.z);
    if (tf < max(tn, 0.0)) { o = vec4(bg, 1.0); return; }
    tn = max(tn, 0.0);
    float dt = length(ext) / steps;
    float t = tn; float prev = 0.0; bool hit = false;
    for (int i = 0; i < 1024; i++) {
      if (t > tf) break;
      vec3 q = (ro + rd * t) / ext + 0.5;
      float v = d(q);
      if (v >= iso) { hit = true; break; }
      prev = v; t += dt;
    }
    if (!hit) { o = vec4(bg, 1.0); return; }
    float a = t - dt, b = t;                       // refine the crossing
    for (int k = 0; k < 6; k++) { float m = 0.5 * (a + b); if (d((ro + rd * m) / ext + 0.5) >= iso) b = m; else a = m; }
    vec3 q = (ro + rd * b) / ext + 0.5;
    vec3 e = 1.0 / vec3(textureSize(vol, 0));
    vec3 g = vec3(d(q + vec3(e.x, 0, 0)) - d(q - vec3(e.x, 0, 0)), d(q + vec3(0, e.y, 0)) - d(q - vec3(0, e.y, 0)),
                  d(q + vec3(0, 0, e.z)) - d(q - vec3(0, 0, e.z)));
    vec3 n = -normalize(g / ext + 1e-9);
    vec3 L = normalize(rot * vec3(0.4, 0.6, 1.0));
    vec3 V = -rd;
    float diff = max(dot(n, L), 0.0), fill = max(dot(n, V), 0.0);
    float spec = pow(max(dot(n, normalize(L + V)), 0.0), 40.0);
    vec3 c = useCol ? ramp(texture(col, q).r) : base;
    vec3 rgb = c * (0.22 + 0.62 * diff + 0.22 * fill) + vec3(0.35) * spec;
    o = vec4(pow(rgb, vec3(1.0 / 1.1)), 1.0);
  }`;
  function mat3FromQuat(q) {
    const [x, y, z, w] = q;
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
            2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
            2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)];
  }
  function qmul(a, b) {
    return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
  }
  function qaxis(ax, ang) { const s = Math.sin(ang / 2); return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)]; }
  function hexrgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }

  function create(canvas, opts = {}) {
    const gl = canvas.getContext('webgl2', {antialias: false, preserveDrawingBuffer: true});
    if (!gl) throw new Error('this browser has no WebGL2: the 3D view needs it');
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const U = n => gl.getUniformLocation(prog, n);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    const tex = (unit, data, shape) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_3D, t);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, shape[2], shape[1], shape[0], 0, gl.RED, gl.UNSIGNED_BYTE, data);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.LINEAR);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
      return t;
    };
    const st = {q: qaxis([1, 0, 0], -0.35), dist: 1.9, iso: 0.5, useCol: false, head: null, texV: null, texC: null,
                base: hexrgb(opts.base || '#b9c7dc'), bg: hexrgb(opts.bg || '#000000'), raf: 0};
    function draw() {
      st.raf = 0;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      if (!st.head) { gl.clearColor(...st.bg, 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
      const sh = st.head.shape, m = Math.max(...sh);
      gl.uniform3f(U('ext'), sh[2] / m, sh[1] / m, sh[0] / m);
      gl.uniformMatrix3fv(U('rot'), false, mat3FromQuat(st.q));
      gl.uniform1f(U('dist'), st.dist); gl.uniform1f(U('iso'), st.iso); gl.uniform1f(U('aspect'), w / h);
      gl.uniform1f(U('steps'), Math.min(900, 3 * m));
      gl.uniform3f(U('base'), ...st.base); gl.uniform3f(U('bg'), ...st.bg);
      gl.uniform1i(U('vol'), 0); gl.uniform1i(U('col'), 1); gl.uniform1i(U('useCol'), st.useCol && !!st.texC);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    const redraw = () => { if (!st.raf) st.raf = requestAnimationFrame(draw); };
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
      drag = [e.clientX, e.clientY];
      const s = 3.0 / Math.max(canvas.clientWidth, 1);
      st.q = qmul(st.q, qmul(qaxis([0, 1, 0], -dx * s), qaxis([1, 0, 0], -dy * s)));
      redraw();
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('wheel', e => { e.preventDefault(); st.dist = Math.min(6, Math.max(0.6, st.dist * Math.exp(e.deltaY * 0.001))); redraw(); }, {passive: false});
    canvas.addEventListener('dblclick', () => { st.q = qaxis([1, 0, 0], -0.35); st.dist = 1.9; redraw(); });
    new ResizeObserver(redraw).observe(canvas);
    return {
      async load(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error((await r.text()) || r.statusText);
        const ab = await r.arrayBuffer();
        const n = new DataView(ab).getUint32(0, true);
        const head = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, n)));
        const size = head.shape[0] * head.shape[1] * head.shape[2];
        if (st.texV) gl.deleteTexture(st.texV);
        if (st.texC) gl.deleteTexture(st.texC);
        st.texV = tex(0, new Uint8Array(ab, 4 + n, size), head.shape);
        st.texC = head.color ? tex(1, new Uint8Array(ab, 4 + n + size, size), head.shape) : null;
        st.head = head;
        redraw();
        return head;
      },
      setLevel(v) { if (!st.head) return; st.iso = Math.min(1, Math.max(0, (v - st.head.min) / (st.head.max - st.head.min))); redraw(); },
      setColor(on) { st.useCol = !!on; redraw(); },
      view(axis) {
        st.q = axis === 'z' ? [0, 0, 0, 1] : axis === 'y' ? qaxis([1, 0, 0], -Math.PI / 2) : qaxis([0, 1, 0], Math.PI / 2);
        redraw();
      },
      snapshot() { draw(); return canvas.toDataURL('image/png'); },
      get head() { return st.head; },
      dispose() { if (st.texV) gl.deleteTexture(st.texV); if (st.texC) gl.deleteTexture(st.texC); },
    };
  }
  window.CelerisVolume = {create};
})();
