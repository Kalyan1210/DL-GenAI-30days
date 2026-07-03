/* Inference Engineering — interactive course. All sims are first-order models. */
"use strict";

/* ---------------- shared helpers ---------------- */
const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x, d = 1) => x.toLocaleString("en-US", { maximumFractionDigits: d });

const C = {
  bg: "#161b28", grid: "#232a3a", ink: "#d7dce6", dim: "#8a93a6",
  teal: "#5eead4", amber: "#f59e0b", indigo: "#818cf8", green: "#34d399",
  red: "#f87171", pink: "#f472b6", blue: "#60a5fa", violet: "#a78bfa",
};
const PALETTE = [C.teal, C.amber, C.indigo, C.green, C.pink, C.blue, C.violet, "#fb923c", "#4ade80", "#e879f9"];

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = +canvas.getAttribute("height");
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: w, H: h };
}

function clear(ctx, W, H) {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
}

function label(ctx, text, x, y, color = C.dim, size = 12, align = "left", mono = true) {
  ctx.fillStyle = color;
  ctx.font = `${size}px ${mono ? "Consolas, Menlo, monospace" : "sans-serif"}`;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
  ctx.textAlign = "left";
}

function roundRect(ctx, x, y, w, h, r, fillColor) {
  ctx.beginPath();
  ctx.roundRect(x, y, Math.max(w, 0.001), h, r);
  ctx.fillStyle = fillColor;
  ctx.fill();
}

/* renderers registry so resize can redraw everything */
const renderers = [];
window.addEventListener("resize", () => renderers.forEach((r) => r()));

/* =========================================================
   MODULE 0 — streaming weights
========================================================= */
(() => {
  const canvas = $("m0-canvas");
  let anim = null, progress = 0; // GB streamed

  const params = () => {
    const p = +$("m0-model").value;              // billions of params
    const gb = p * 2;                            // FP16 bytes
    const bw = +$("m0-bw").value * 1000;         // GB/s
    return { p, gb, bw };
  };

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const { gb, bw } = params();
    const frac = clamp(progress / gb, 0, 1);

    // HBM pipe
    const bx = 40, bw_ = W - 80, by = 70, bh = 36;
    roundRect(ctx, bx, by, bw_, bh, 8, "#0b0e14");
    ctx.strokeStyle = C.grid; ctx.strokeRect(bx, by, bw_, bh);
    roundRect(ctx, bx, by, bw_ * frac, bh, 8, C.teal);

    // flowing chunks
    if (anim) {
      ctx.fillStyle = "rgba(94,234,212,.35)";
      const t = performance.now() / 200;
      for (let i = 0; i < 14; i++) {
        const px = bx + ((i / 14 + (t % 1) / 14) % 1) * bw_ * frac;
        ctx.fillRect(px, by + 6, 3, bh - 12);
      }
    }
    label(ctx, "HBM  →  compute units", bx, by - 12, C.dim, 12);
    label(ctx, `${fmt(progress, 1)} / ${fmt(gb, 0)} GB of weights streamed`, bx, by + bh + 22, C.ink, 13);

    const ms = (progress / bw) * 1000;
    label(ctx, `elapsed: ${fmt(ms, 2)} ms`, W - 40, by + bh + 22, C.amber, 13, "right");

    if (frac >= 1) {
      label(ctx, `● token emitted — this is your floor for TPOT at batch=1`, bx, by + bh + 48, C.green, 13);
    }
  }

  function readout() {
    const { p, gb, bw } = params();
    const ms = (gb / bw) * 1000;
    $("m0-read").innerHTML =
      `${p}B params × 2 bytes (FP16) = <b>${fmt(gb, 0)} GB</b> per token · ` +
      `at ${fmt(bw / 1000, 2)} TB/s → TPOT floor <b>${fmt(ms, 1)} ms</b> · ` +
      `max <b>${fmt(1000 / ms, 1)} tok/s</b> for one user. ` +
      (ms > 50 ? `<span class="warn">Slower than human reading speed — unusable without the techniques ahead.</span>`
               : `<span class="ok">Interactive.</span>`);
  }

  function run() {
    cancelAnimationFrame(anim);
    progress = 0;
    const { gb } = params();
    const dur = 1800; // ms of animation regardless of real time
    const t0 = performance.now();
    const step = () => {
      progress = clamp(((performance.now() - t0) / dur) * gb, 0, gb);
      draw();
      if (progress < gb) anim = requestAnimationFrame(step);
      else anim = null;
    };
    anim = requestAnimationFrame(step);
  }

  $("m0-go").onclick = run;
  $("m0-bw").oninput = () => { $("m0-bw-v").textContent = fmt(+$("m0-bw").value, 2) + " TB/s"; draw(); readout(); };
  $("m0-model").onchange = () => { progress = 0; draw(); readout(); };
  renderers.push(draw);
  draw(); readout();
})();

/* =========================================================
   MODULE 1 — prefill vs decode
========================================================= */
(() => {
  const canvas = $("m1-canvas");
  let state = { phase: "idle", tick: 0, hist: [] }; // hist: [{comp, bw}]
  let timer = null;

  const P = () => +$("m1-p").value;
  const G = () => +$("m1-g").value;

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const p = P(), g = G(), total = p + g;
    const pad = 30, availW = W - pad * 2;
    const cell = Math.min(24, availW / total - 2);
    const y0 = 44;

    label(ctx, "token stream (■ prompt   ■ generated)", pad, 26, C.dim, 12);
    for (let i = 0; i < total; i++) {
      const x = pad + i * (cell + 2);
      let color = "#0b0e14";
      if (state.phase !== "idle") {
        if (i < p && (state.phase !== "prefillWait")) color = C.indigo;
        if (i >= p && i - p < state.tick - 1) color = C.teal;
        if (state.phase === "decode" && i - p === state.tick - 1 && i >= p) color = C.amber;
      }
      roundRect(ctx, x, y0, cell, cell, 3, color);
      ctx.strokeStyle = C.grid;
      ctx.strokeRect(x, y0, cell, cell);
    }

    // utilization history chart
    const cy = y0 + cell + 44, ch = H - cy - 40;
    label(ctx, "GPU compute utilization", pad, cy - 8, C.indigo, 12);
    label(ctx, "memory bandwidth utilization", pad + 240, cy - 8, C.teal, 12);
    ctx.strokeStyle = C.grid;
    ctx.strokeRect(pad, cy, availW, ch);
    const n = Math.max(state.hist.length, 1);
    const dx = availW / Math.max(total + 2, n);
    for (const [key, color] of [["comp", C.indigo], ["bw", C.teal]]) {
      ctx.beginPath();
      state.hist.forEach((h, i) => {
        const x = pad + i * dx, y = cy + ch - h[key] * ch;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
    }
    label(ctx, "100%", pad - 4, cy + 10, C.dim, 10, "right");
    label(ctx, "0%", pad - 4, cy + ch, C.dim, 10, "right");
    if (state.phase === "prefill") label(ctx, "PREFILL: all prompt tokens in one GEMM", pad, cy + ch + 24, C.indigo, 12);
    if (state.phase === "decode") label(ctx, `DECODE: step ${state.tick}/${g} — one token per full weight read`, pad, cy + ch + 24, C.amber, 12);
    if (state.phase === "done") label(ctx, "done — note compute spiked once, bandwidth stayed pinned throughout decode", pad, cy + ch + 24, C.green, 12);
  }

  function run() {
    clearInterval(timer);
    state = { phase: "prefill", tick: 0, hist: [] };
    const g = G();
    let step = 0;
    timer = setInterval(() => {
      step++;
      if (step === 1) {
        state.phase = "prefill";
        state.hist.push({ comp: 0.95, bw: 0.55 });
      } else if (step === 2) {
        state.phase = "decode"; state.tick = 0;
        state.hist.push({ comp: 0.9, bw: 0.6 });
      } else if (state.tick < g) {
        state.tick++;
        state.hist.push({ comp: 0.04 + Math.random() * 0.03, bw: 0.93 + Math.random() * 0.05 });
      } else {
        state.phase = "done";
        clearInterval(timer);
      }
      draw(); readout();
    }, 240);
  }

  function readout() {
    const p = P(), g = G();
    const prefillFlops = p, decodeSteps = g;
    $("m1-read").innerHTML =
      `prefill: <b>${p} tokens in 1 pass</b> (compute-bound, sets TTFT) · ` +
      `decode: <b>${g} tokens in ${decodeSteps} sequential passes</b> (bandwidth-bound, sets TPOT) · ` +
      `weight reads: prefill <b>1×</b>, decode <b>${g}×</b> — decode reads ${g}× the bytes to do ${(g / p) < 1 ? "less" : "comparable"} math.`;
  }

  $("m1-p").oninput = () => { $("m1-p-v").textContent = P(); draw(); readout(); };
  $("m1-g").oninput = () => { $("m1-g-v").textContent = G(); draw(); readout(); };
  $("m1-run").onclick = run;
  renderers.push(draw);
  draw(); readout();
})();

/* =========================================================
   MODULE 2 — roofline
========================================================= */
(() => {
  const canvas = $("m2-canvas");
  const GPUS = {
    a100: { name: "A100", tf: 312, bw: 2.0 },
    h100: { name: "H100", tf: 989, bw: 3.35 },
    b200: { name: "B200", tf: 2250, bw: 8.0 },
    rtx4090: { name: "RTX 4090", tf: 165, bw: 1.0 },
  };
  const batch = () => Math.round(Math.pow(2, +$("m2-b").value));

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const g = GPUS[$("m2-gpu").value];
    const padL = 64, padR = 30, padT = 24, padB = 44;
    const pw = W - padL - padR, ph = H - padT - padB;

    // log axes: x AI 0.5 .. 16384 ; y TFLOPs 0.1 .. 4000
    const xmin = Math.log10(0.5), xmax = Math.log10(16384);
    const ymin = Math.log10(0.1), ymax = Math.log10(4000);
    const X = (ai) => padL + ((Math.log10(ai) - xmin) / (xmax - xmin)) * pw;
    const Y = (tf) => padT + ph - ((Math.log10(tf) - ymin) / (ymax - ymin)) * ph;

    // grid
    ctx.strokeStyle = C.grid;
    for (let e = 0; e <= 4; e++) {
      const ai = Math.pow(10, e);
      ctx.beginPath(); ctx.moveTo(X(ai), padT); ctx.lineTo(X(ai), padT + ph); ctx.stroke();
      label(ctx, "10^" + e, X(ai), padT + ph + 18, C.dim, 11, "center");
    }
    for (let e = 0; e <= 3; e++) {
      const tf = Math.pow(10, e);
      ctx.beginPath(); ctx.moveTo(padL, Y(tf)); ctx.lineTo(padL + pw, Y(tf)); ctx.stroke();
      label(ctx, fmt(tf, 0), padL - 8, Y(tf) + 4, C.dim, 11, "right");
    }
    label(ctx, "arithmetic intensity (FLOPs / byte)", padL + pw / 2, H - 8, C.dim, 12, "center");
    ctx.save(); ctx.translate(16, padT + ph / 2); ctx.rotate(-Math.PI / 2);
    label(ctx, "attainable TFLOP/s", 0, 0, C.dim, 12, "center"); ctx.restore();

    // roofline
    const ridge = g.tf / g.bw;
    ctx.beginPath();
    ctx.moveTo(X(0.5), Y(0.5 * g.bw));
    ctx.lineTo(X(ridge), Y(g.tf));
    ctx.lineTo(X(16384), Y(g.tf));
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
    label(ctx, `memory roof (${g.bw} TB/s)`, X(2.2), Y(2.2 * g.bw) - 10, C.teal, 12);
    label(ctx, `compute roof (${g.tf} TFLOPS)`, X(2500), Y(g.tf) - 10, C.indigo, 12, "center");
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(ridge), padT + ph); ctx.lineTo(X(ridge), Y(g.tf)); ctx.strokeStyle = C.dim; ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, `ridge ≈ ${fmt(ridge, 0)}`, X(ridge), padT + ph - 6, C.amber, 11, "center");

    // operating points
    const b = batch();
    const aiDecode = b;                     // ~B FLOPs per byte
    const perfDecode = Math.min(aiDecode * g.bw, g.tf);
    const aiPrefill = 2048;
    const perfPrefill = Math.min(aiPrefill * g.bw, g.tf);

    ctx.beginPath(); ctx.arc(X(aiPrefill), Y(perfPrefill), 7, 0, 7); ctx.fillStyle = C.indigo; ctx.fill();
    label(ctx, "prefill (2048-tok prompt)", X(aiPrefill), Y(perfPrefill) + 24, C.indigo, 12, "center");

    ctx.beginPath(); ctx.arc(X(aiDecode), Y(perfDecode), 8, 0, 7); ctx.fillStyle = C.amber; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.stroke();
    label(ctx, `decode B=${b}`, X(aiDecode), Y(perfDecode) - 14, C.amber, 13, "center");

    return { g, b, perfDecode, ridge };
  }

  function readout() {
    const r = draw();
    const pct = (r.perfDecode / r.g.tf) * 100;
    const bound = r.b < r.ridge ? `<span class="warn">memory-bandwidth-bound</span>` : `<span class="ok">compute-bound</span>`;
    $("m2-read").innerHTML =
      `${r.g.name}: decode at batch <b>${r.b}</b> attains <b>${fmt(r.perfDecode, 0)} TFLOP/s</b> ` +
      `(${fmt(pct, 1)}% of peak) — ${bound}. ` +
      `Batching ×2 doubles attainable throughput until batch ≈ <b>${fmt(r.ridge, 0)}</b> (the ridge point); after that, more batch only adds latency.`;
  }

  $("m2-b").oninput = () => { $("m2-b-v").textContent = batch(); readout(); };
  $("m2-gpu").onchange = readout;
  renderers.push(readout);
  readout();
})();

/* =========================================================
   MODULE 3 — KV cache
========================================================= */
(() => {
  const canvas = $("m3-canvas"), mem = $("m3-mem");
  const PRESETS = {
    llama8b: { L: 32, kv: 8, hd: 128, wGB: 16 },
    llama70b: { L: 80, kv: 8, hd: 128, wGB: 140 },
    mha70b: { L: 80, kv: 64, hd: 128, wGB: 140 },
  };
  let animStep = 0, timer = null;

  const cfg = () => PRESETS[$("m3-preset").value];
  const seq = () => Math.round(Math.pow(2, +$("m3-s").value));
  const bt = () => +$("m3-bt").value;
  const dt = () => +$("m3-dt").value;

  function kvPerTokenBytes() {
    const c = cfg();
    return 2 * c.L * c.kv * c.hd * dt();
  }

  function drawAttn() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const N = 18;
    const pad = 44;
    const cell = Math.min((W - pad - 16) / N, (H - pad - 30) / N);
    label(ctx, "attention matrix (rows = query step, cols = keys)", pad, 20, C.dim, 11);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        const x = pad + j * cell, y = 34 + i * cell;
        let color = "#0b0e14";
        if (i < animStep) color = "rgba(52,211,153,.55)";           // cached history
        if (i === animStep) color = j === i ? C.amber : "rgba(52,211,153,.85)";
        roundRect(ctx, x, y, cell - 1.5, cell - 1.5, 2, color);
      }
    }
    label(ctx, "green = K/V read from cache", pad, 34 + N * cell + 14, C.green, 11);
    label(ctx, "orange = new K/V this step", pad, 34 + N * cell + 28, C.amber, 11);
  }

  function drawMem() {
    const { ctx, W, H } = fit(mem);
    clear(ctx, W, H);
    const c = cfg();
    const HBM = 80; // GB, one H100
    const kvGB = (kvPerTokenBytes() * seq() * bt()) / 1e9;
    const wGB = c.wGB * (2 / 2); // weights always FP16 here
    const total = wGB + kvGB;

    const bx = 60, bw = W - 100, by = 40, bh = H - 110;
    label(ctx, "one H100 (80 GB HBM)", bx, 26, C.dim, 12);
    ctx.strokeStyle = C.grid; ctx.strokeRect(bx, by, bw, bh);

    const scale = bh / Math.max(HBM, total);
    const wH = wGB * scale, kvH = Math.min(kvGB * scale, bh * 2);
    roundRect(ctx, bx + 2, by + bh - wH, bw - 4, wH, 3, C.indigo);
    roundRect(ctx, bx + 2, by + bh - wH - kvH, bw - 4, kvH, 3, total > HBM ? C.red : C.teal);
    // 80GB line
    const yl = by + bh - HBM * scale;
    ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(bx - 8, yl); ctx.lineTo(bx + bw + 8, yl);
    ctx.strokeStyle = C.amber; ctx.stroke(); ctx.setLineDash([]);
    label(ctx, "80 GB", bx - 8, yl + 4, C.amber, 11, "right");

    label(ctx, `weights ${fmt(wGB, 0)} GB`, bx + bw / 2, by + bh - wH / 2 + 4, "#fff", 12, "center");
    if (kvH > 14) label(ctx, `KV cache ${fmt(kvGB, 1)} GB`, bx + bw / 2, by + bh - wH - kvH / 2 + 4, "#04211c", 12, "center");
    label(ctx, total > HBM ? "✗ DOES NOT FIT — OOM / must evict, page, or shrink" : "✓ fits",
      bx, by + bh + 28, total > HBM ? C.red : C.green, 13);
    label(ctx, `KV per token: ${fmt(kvPerTokenBytes() / 1024, 1)} KiB`, bx, by + bh + 50, C.dim, 12);
  }

  function readout() {
    const kvGB = (kvPerTokenBytes() * seq() * bt()) / 1e9;
    const c = cfg();
    $("m3-read").innerHTML =
      `2 × ${c.L} layers × ${c.kv} KV heads × ${c.hd} dim × ${dt()} B = ` +
      `<b>${fmt(kvPerTokenBytes() / 1024, 1)} KiB/token</b> · × ${fmt(seq(), 0)} tokens × batch ${bt()} = ` +
      `<b>${fmt(kvGB, 1)} GB</b> of cache — vs <b>${cfg().wGB} GB</b> of weights. ` +
      `Every decode step must also <em>read</em> the sequence's cache: +${fmt(kvPerTokenBytes() * seq() / 1e6, 0)} MB/step/seq of bandwidth.`;
  }

  function animate() {
    clearInterval(timer);
    animStep = 0;
    timer = setInterval(() => {
      animStep++;
      if (animStep > 18) { clearInterval(timer); animStep = 18; }
      drawAttn();
    }, 150);
  }

  const refresh = () => { drawAttn(); drawMem(); readout(); };
  $("m3-preset").onchange = refresh;
  $("m3-s").oninput = () => { $("m3-s-v").textContent = fmt(seq(), 0); refresh(); };
  $("m3-bt").oninput = () => { $("m3-bt-v").textContent = bt(); refresh(); };
  $("m3-dt").onchange = refresh;
  $("m3-anim").onclick = animate;
  renderers.push(refresh);
  animStep = 6;
  refresh();
})();

/* =========================================================
   MODULE 4 — continuous batching
========================================================= */
(() => {
  const canvas = $("m4-canvas");
  let timer = null, sim = null, t = 0;

  function makeStream(T, rate) {
    // deterministic-ish stream: poisson arrivals, geometric-ish lengths
    const reqs = [];
    let id = 0;
    for (let step = 0; step < T; step++) {
      let n = 0;
      // poisson via thinning
      let p = Math.exp(-rate), cum = p, u = Math.random();
      while (u > cum) { n++; p = p * rate / n; cum += p; }
      for (let i = 0; i < n; i++) {
        reqs.push({ id: id++, arrive: step, len: 3 + Math.floor(Math.random() * 22) });
      }
    }
    return reqs;
  }

  function simulate(T, rate, S) {
    const stream = makeStream(T, rate);
    // ---- static batching ----
    const st = { grid: [], queue: [], done: [], batchEnd: 0, batch: [] };
    const cb = { grid: [], queue: [], done: [], slots: new Array(S).fill(null) };
    for (let k = 0; k < S; k++) { st.grid.push([]); cb.grid.push([]); }

    let si = 0;
    const stQueue = [], cbQueue = [];
    for (let step = 0; step < T; step++) {
      while (si < stream.length && stream[si].arrive === step) {
        stQueue.push({ ...stream[si] });
        cbQueue.push({ ...stream[si] });
        si++;
      }
      // static: if current batch finished, load a new one
      if (st.batch.length === 0 || st.batch.every((r) => r.doneAt !== undefined)) {
        st.batch.forEach((r) => st.done.push(r));
        st.batch = [];
        while (st.batch.length < S && stQueue.length) {
          const r = stQueue.shift();
          r.remaining = r.len; r.slot = st.batch.length; r.start = step;
          st.batch.push(r);
        }
      }
      const maxRem = Math.max(0, ...st.batch.filter((r) => r.doneAt === undefined).map((r) => r.remaining));
      for (let k = 0; k < S; k++) {
        const r = st.batch[k];
        if (!r) { st.grid[k].push(null); continue; }
        if (r.doneAt !== undefined) { st.grid[k].push({ id: r.id, idle: true }); continue; }
        r.remaining--;
        st.grid[k].push({ id: r.id });
        if (r.remaining <= 0) { r.doneAt = step; r.latency = step - r.arrive; }
      }
      // continuous
      for (let k = 0; k < S; k++) {
        if (!cb.slots[k] && cbQueue.length) {
          const r = cbQueue.shift();
          r.remaining = r.len; r.start = step;
          cb.slots[k] = r;
        }
        const r = cb.slots[k];
        if (!r) { cb.grid[k].push(null); continue; }
        r.remaining--;
        cb.grid[k].push({ id: r.id });
        if (r.remaining <= 0) {
          r.doneAt = step; r.latency = step - r.arrive;
          cb.done.push(r); cb.slots[k] = null;
        }
      }
    }
    st.batch.forEach((r) => { if (r.doneAt !== undefined) st.done.push(r); });
    return { st, cb, T, S };
  }

  function draw(upto) {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    if (!sim) { label(ctx, "press Run simulation", 30, 40, C.dim, 14); return; }
    const { st, cb, T, S } = sim;
    const pad = 30, colW = (W - pad * 2) / T;
    const panelH = (H - 110) / 2;
    const rowH = Math.min(26, (panelH - 30) / S);

    const panels = [
      { name: "STATIC BATCHING", data: st.grid, y: 46, done: st.done },
      { name: "CONTINUOUS BATCHING", data: cb.grid, y: 46 + panelH + 30, done: cb.done },
    ];
    for (const p of panels) {
      label(ctx, p.name, pad, p.y - 10, C.ink, 13);
      for (let k = 0; k < S; k++) {
        for (let x = 0; x < Math.min(upto, T); x++) {
          const cellVal = p.data[k][x];
          const cx = pad + x * colW, cy = p.y + k * rowH;
          let color = "#0b0e14";
          if (cellVal) color = cellVal.idle ? "#39404f" : PALETTE[cellVal.id % PALETTE.length];
          ctx.fillStyle = color;
          ctx.fillRect(cx, cy, Math.max(colW - 0.6, 0.5), rowH - 2);
        }
      }
      const doneN = p.done.filter((r) => r.doneAt < upto).length;
      const lats = p.done.filter((r) => r.doneAt < upto).map((r) => r.latency);
      const mean = lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : 0;
      label(ctx, `completed: ${doneN}   mean latency: ${fmt(mean, 1)} steps`, W - pad, p.y - 10, C.teal, 12, "right");
    }
    label(ctx, "time (decode steps) →   grey = slot held by a finished request (waste)", pad, H - 12, C.dim, 11);
  }

  function readout() {
    if (!sim) return;
    const done = (d) => d.filter((r) => r.doneAt !== undefined);
    const meanLat = (d) => { const l = done(d).map((r) => r.latency); return l.length ? l.reduce((a, b) => a + b) / l.length : 0; };
    const stN = done(sim.st.done).length, cbN = done(sim.cb.done).length;
    $("m4-read").innerHTML =
      `same request stream → static: <b>${stN} done</b>, mean latency <b>${fmt(meanLat(sim.st.done), 1)}</b> steps · ` +
      `continuous: <b class="ok">${cbN} done</b>, mean latency <b class="ok">${fmt(meanLat(sim.cb.done), 1)}</b> steps · ` +
      `throughput gain ≈ <b>${stN ? fmt(cbN / stN, 2) : "∞"}×</b>. Grey cells are pure waste: a slot pinned by a request that already finished.`;
  }

  function run() {
    clearInterval(timer);
    const T = 120;
    sim = simulate(T, +$("m4-r").value, +$("m4-sl").value);
    t = 0;
    timer = setInterval(() => {
      t += 2;
      if (t >= T) { t = T; clearInterval(timer); readout(); }
      draw(t);
    }, 40);
  }

  $("m4-run").onclick = run;
  $("m4-stop").onclick = () => clearInterval(timer);
  $("m4-r").oninput = () => $("m4-r-v").textContent = fmt(+$("m4-r").value, 2) + " req/step";
  $("m4-sl").oninput = () => $("m4-sl-v").textContent = $("m4-sl").value;
  renderers.push(() => draw(t));
  draw(0);
})();

/* =========================================================
   MODULE 5 — PagedAttention
========================================================= */
(() => {
  const canvas = $("m5-canvas");
  const BLOCKS = 96;          // physical blocks per allocator
  const BLOCK_TOK = 16;       // tokens per block
  const MAXLEN_BLOCKS = 16;   // naive reserves this many contiguous blocks
  let naive = [], paged = [], nextId = 0, auto = null;

  // naive: array of {id, startBlock, reserved, usedBlocks}
  // paged: array of {id, blocks: [physIdx], usedTokensInLast}

  function naiveOwner(bi) {
    for (const r of naive) if (bi >= r.start && bi < r.start + MAXLEN_BLOCKS) return r;
    return null;
  }

  function spawn() {
    const actualBlocks = 1 + Math.floor(Math.random() * (MAXLEN_BLOCKS - 1) * 0.6); // usually well under max
    // naive contiguous fit
    let placed = false;
    for (let s = 0; s + MAXLEN_BLOCKS <= BLOCKS; s++) {
      let free = true;
      for (let b = s; b < s + MAXLEN_BLOCKS; b++) if (naiveOwner(b)) { free = false; break; }
      if (free) { naive.push({ id: nextId, start: s, used: actualBlocks }); placed = true; break; }
    }
    // paged fit
    const freeBlocks = [];
    const owned = new Set(paged.flatMap((r) => r.blocks));
    for (let b = 0; b < BLOCKS; b++) if (!owned.has(b)) freeBlocks.push(b);
    let pagedPlaced = false;
    if (freeBlocks.length >= actualBlocks) {
      paged.push({ id: nextId, blocks: freeBlocks.slice(0, actualBlocks) });
      pagedPlaced = true;
    }
    nextId++;
    draw(); readout(placed, pagedPlaced);
  }

  function kill() {
    if (!naive.length && !paged.length) return;
    const ids = [...new Set([...naive.map((r) => r.id), ...paged.map((r) => r.id)])];
    const victim = ids[Math.floor(Math.random() * ids.length)];
    naive = naive.filter((r) => r.id !== victim);
    paged = paged.filter((r) => r.id !== victim);
    draw(); readout();
  }

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const cols = 24, rows = BLOCKS / cols;
    const pad = 30, gw = W - pad * 2;
    const cell = gw / cols;
    const cellH = Math.min(cell, 34);

    const drawGrid = (y0, title, getCell) => {
      label(ctx, title, pad, y0 - 10, C.ink, 13);
      for (let b = 0; b < BLOCKS; b++) {
        const x = pad + (b % cols) * cell, y = y0 + Math.floor(b / cols) * cellH;
        const info = getCell(b);
        roundRect(ctx, x, y, cell - 2, cellH - 2, 3, info.color);
        if (info.hatch) {
          ctx.strokeStyle = "rgba(255,255,255,.25)";
          ctx.beginPath();
          ctx.moveTo(x + 2, y + cellH - 4); ctx.lineTo(x + cell - 4, y + 2);
          ctx.stroke();
        }
        ctx.strokeStyle = C.grid; ctx.strokeRect(x, y, cell - 2, cellH - 2);
      }
    };

    drawGrid(46, "NAIVE: contiguous max-length reservation per request", (b) => {
      const r = naiveOwner(b);
      if (!r) return { color: "#0b0e14" };
      const idx = b - r.start;
      if (idx < r.used) return { color: PALETTE[r.id % PALETTE.length] };
      return { color: "rgba(248,113,113,.28)", hatch: true }; // reserved, unused
    });

    const y2 = 46 + (rows * cellH) + 52;
    const ownedMap = new Map();
    paged.forEach((r) => r.blocks.forEach((b) => ownedMap.set(b, r.id)));
    drawGrid(y2, "PAGED: 16-token blocks allocated on demand, block table per request", (b) => {
      if (ownedMap.has(b)) return { color: PALETTE[ownedMap.get(b) % PALETTE.length] };
      return { color: "#0b0e14" };
    });

    label(ctx, "hatched red = reserved but never used (internal fragmentation)", pad, y2 - 26, C.red, 11);
  }

  function readout(naiveOk, pagedOk) {
    const naiveUsed = naive.reduce((a, r) => a + r.used, 0);
    const naiveRes = naive.length * MAXLEN_BLOCKS;
    const pagedUsed = paged.reduce((a, r) => a + r.blocks.length, 0);
    let msg =
      `naive: <b>${naive.length} reqs</b>, ${naiveRes}/${BLOCKS} blocks reserved, only ${naiveUsed} hold data ` +
      `(<span class="warn">${naiveRes ? fmt((naiveUsed / naiveRes) * 100, 0) : 0}% useful</span>) · ` +
      `paged: <b class="ok">${paged.length} reqs</b>, ${pagedUsed}/${BLOCKS} blocks, ` +
      `<span class="ok">~100% useful</span>.`;
    if (naiveOk === false && pagedOk === true) msg += ` <span class="err">Naive just OOM'd — paged still had room!</span>`;
    else if (naiveOk === false && pagedOk === false) msg += ` <span class="err">Both allocators full.</span>`;
    $("m5-read").innerHTML = msg;
  }

  $("m5-spawn").onclick = spawn;
  $("m5-kill").onclick = kill;
  $("m5-reset").onclick = () => { naive = []; paged = []; nextId = 0; clearInterval(auto); auto = null; draw(); readout(); };
  $("m5-auto").onclick = () => {
    if (auto) { clearInterval(auto); auto = null; return; }
    auto = setInterval(() => (Math.random() < 0.65 ? spawn() : kill()), 500);
  };
  renderers.push(draw);
  draw(); readout();
})();

/* =========================================================
   MODULE 6 — quantization
========================================================= */
(() => {
  const canvas = $("m6-canvas");
  const N = 256, GROUP = 32;
  let weights = [];

  function resample() {
    weights = [];
    const useOut = $("m6-out").checked;
    for (let i = 0; i < N; i++) {
      // gaussian via Box-Muller
      const u = Math.random(), v = Math.random();
      let w = Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v) * 0.5;
      weights.push(w);
    }
    if (useOut) {
      for (let k = 0; k < 4; k++) {
        weights[Math.floor(Math.random() * N)] = (Math.random() < 0.5 ? -1 : 1) * (3.5 + Math.random() * 2);
      }
    }
  }

  function quantize() {
    const bits = +$("m6-b").value;
    const levels = Math.pow(2, bits - 1) - 1; // symmetric signed
    const perGroup = $("m6-grp").checked;
    const q = new Array(N), scales = [];
    const gsize = perGroup ? GROUP : N;
    for (let g = 0; g < N / gsize; g++) {
      let amax = 1e-9;
      for (let i = g * gsize; i < (g + 1) * gsize; i++) amax = Math.max(amax, Math.abs(weights[i]));
      const s = amax / levels;
      scales.push(s);
      for (let i = g * gsize; i < (g + 1) * gsize; i++) {
        q[i] = clamp(Math.round(weights[i] / s), -levels, levels) * s;
      }
    }
    return { q, scales, bits, gsize };
  }

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const { q, scales, bits, gsize } = quantize();
    const pad = 46, pw = W - pad - 20;
    const topH = H * 0.58, botY = topH + 34, botH = H - botY - 26;
    const vmax = Math.max(2.2, ...weights.map(Math.abs)) * 1.08;
    const X = (i) => pad + (i / (N - 1)) * pw;
    const Ytop = (v) => 20 + (topH - 40) * (0.5 - v / (2 * vmax));

    // group boundaries + grid levels for first group's scale (visual aid)
    ctx.strokeStyle = C.grid;
    for (let g = 1; g < N / gsize; g++) {
      const x = X(g * gsize);
      ctx.beginPath(); ctx.moveTo(x, 16); ctx.lineTo(x, topH - 14); ctx.stroke();
    }
    // zero line
    ctx.strokeStyle = "#39404f";
    ctx.beginPath(); ctx.moveTo(pad, Ytop(0)); ctx.lineTo(pad + pw, Ytop(0)); ctx.stroke();

    // per-group quant grid levels (faint)
    scales.forEach((s, g) => {
      const x0 = X(g * gsize), x1 = X(Math.min((g + 1) * gsize, N - 1));
      const levels = Math.pow(2, bits - 1) - 1;
      const step = Math.max(1, Math.floor(levels / 8));
      ctx.strokeStyle = "rgba(129,140,248,.18)";
      for (let l = -levels; l <= levels; l += step) {
        const y = Ytop(l * s);
        if (y < 16 || y > topH - 14) continue;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      }
    });

    // weights: original dot + quantized tick + connector
    let mse = 0;
    for (let i = 0; i < N; i++) {
      const x = X(i), yo = Ytop(weights[i]), yq = Ytop(q[i]);
      mse += (weights[i] - q[i]) ** 2;
      ctx.strokeStyle = "rgba(245,158,11,.5)";
      ctx.beginPath(); ctx.moveTo(x, yo); ctx.lineTo(x, yq); ctx.stroke();
      ctx.fillStyle = C.teal; ctx.fillRect(x - 1.2, yo - 1.2, 2.4, 2.4);
      ctx.fillStyle = C.indigo; ctx.fillRect(x - 2, yq - 1, 4, 2);
    }
    mse /= N;
    label(ctx, "● original (FP16)    ▬ quantized    | rounding error    faint lines = quant grid", pad, 14, C.dim, 11);

    // error plot
    label(ctx, "per-weight error", pad, botY - 8, C.dim, 11);
    ctx.strokeStyle = C.grid; ctx.strokeRect(pad, botY, pw, botH);
    const emax = Math.max(1e-6, ...weights.map((w, i) => Math.abs(w - q[i])));
    for (let i = 0; i < N; i++) {
      const e = Math.abs(weights[i] - q[i]);
      const h = (e / emax) * (botH - 4);
      ctx.fillStyle = e > emax * 0.5 ? C.red : C.amber;
      ctx.fillRect(X(i) - 1, botY + botH - h, 2.2, h);
    }
    return mse;
  }

  function readout() {
    const mse = draw();
    const bits = +$("m6-b").value;
    const comp = 16 / bits;
    const grp = $("m6-grp").checked;
    $("m6-read").innerHTML =
      `INT${bits}${grp ? " + per-group scales" : " (one scale for all)"}: ` +
      `MSE <b>${mse.toExponential(2)}</b> · model size <b>÷${fmt(comp, 1)}</b> vs FP16 · ` +
      `memory-bound decode speedup ≈ <b class="ok">${fmt(comp, 1)}×</b>` +
      ($("m6-out").checked && !grp
        ? ` · <span class="err">outliers are stretching the single scale — everything else is being crushed. Enable per-group.</span>`
        : "");
  }

  $("m6-b").oninput = () => { $("m6-b-v").textContent = $("m6-b").value; readout(); };
  $("m6-grp").onchange = readout;
  $("m6-out").onchange = () => { resample(); readout(); };
  $("m6-new").onclick = () => { resample(); readout(); };
  renderers.push(readout);
  resample(); readout();
})();

/* =========================================================
   MODULE 7 — speculative decoding
========================================================= */
(() => {
  const canvas = $("m7-canvas");
  let timer = null;
  let lanes = { base: [], spec: [] }; // arrays of {t, w, accepted}
  let clockBase = 0, clockSpec = 0, running = false;

  const K = () => +$("m7-k").value;
  const A = () => +$("m7-a").value;
  const Cost = () => +$("m7-c").value / 100;

  function theory() {
    const k = K(), a = A(), c = Cost();
    const eTok = (1 - Math.pow(a, k + 1)) / (1 - a);
    const roundCost = 1 + k * c;
    return { eTok, speedup: eTok / roundCost };
  }

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const pad = 30, pw = W - pad * 2;
    const TMAX = 24; // time units shown
    const X = (t) => pad + (t / TMAX) * pw;

    const lane = (y, name, toks, clock, color) => {
      label(ctx, name, pad, y - 12, C.ink, 13);
      ctx.strokeStyle = C.grid;
      ctx.beginPath(); ctx.moveTo(pad, y + 26); ctx.lineTo(pad + pw, y + 26); ctx.stroke();
      for (const tok of toks) {
        const x = X(tok.t);
        if (x > pad + pw) continue;
        roundRect(ctx, x, y, Math.max((pw / TMAX) * tok.w - 2, 3), 22, 3, tok.color || color);
      }
      label(ctx, `t = ${fmt(clock, 1)} target-passes · ${toks.filter((x) => x.tok).length} tokens`, W - pad, y - 12, C.teal, 12, "right");
    };

    lane(64, "BASELINE — one target pass per token", lanes.base, clockBase, C.indigo);
    lane(170, "SPECULATIVE — draft k, verify in one pass", lanes.spec, clockSpec, C.teal);

    // legend + acceptance viz
    label(ctx, "■ target pass", pad, 250, C.indigo, 12);
    label(ctx, "■ draft tokens", pad + 120, 250, C.violet, 12);
    label(ctx, "■ accepted", pad + 240, 250, C.green, 12);
    label(ctx, "■ rejected (wasted draft)", pad + 340, 250, C.red, 12);

    const th = theory();
    label(ctx, `theory: E[tokens/round] = (1−α^(k+1))/(1−α) = ${fmt(th.eTok, 2)}   →   expected speedup ≈ ${fmt(th.speedup, 2)}×`,
      pad, 290, C.amber, 13);

    const bTok = lanes.base.filter((x) => x.tok).length;
    const sTok = lanes.spec.filter((x) => x.tok).length;
    if (clockBase > 0 && bTok > 0 && sTok > 0) {
      const su = (sTok / Math.max(clockSpec, 0.01)) / (bTok / clockBase);
      label(ctx, `measured this run: ${fmt(su, 2)}×`, pad, 316, C.green, 13);
    }
  }

  function run() {
    clearInterval(timer);
    lanes = { base: [], spec: [] };
    clockBase = 0; clockSpec = 0;
    const k = K(), a = A(), c = Cost();
    const TMAX = 24;
    timer = setInterval(() => {
      // baseline: one pass -> one token
      if (clockBase + 1 <= TMAX) {
        lanes.base.push({ t: clockBase, w: 1, tok: true, color: C.indigo });
        clockBase += 1;
      }
      // speculative round
      if (clockSpec + k * c + 1 <= TMAX) {
        let t = clockSpec;
        let accepted = 0;
        while (accepted < k && Math.random() < a) accepted++;
        for (let i = 0; i < k; i++) {
          lanes.spec.push({ t, w: c, tok: false, color: i < accepted ? C.violet : "rgba(248,113,113,.6)" });
          t += c;
        }
        // verify pass: accepted tokens + 1 bonus token
        for (let i = 0; i < accepted + 1; i++) {
          lanes.spec.push({ t: t + (i * 1) / (accepted + 1), w: 1 / (accepted + 1), tok: true, color: i < accepted ? C.green : C.teal });
        }
        clockSpec = t + 1;
      }
      draw();
      if (clockBase + 1 > TMAX && clockSpec + k * c + 1 > TMAX) clearInterval(timer);
    }, 260);
  }

  function readout() {
    const th = theory();
    $("m7-read").innerHTML =
      `k=${K()}, α=${fmt(A(), 2)}, draft cost ${fmt(Cost() * 100, 0)}% → ` +
      `<b>${fmt(th.eTok, 2)} tokens per target pass</b>, net speedup <b class="ok">${fmt(th.speedup, 2)}×</b>. ` +
      (th.speedup < 1 ? `<span class="err">Below 1× — the draft is too slow or too wrong; speculation is hurting.</span>`
                      : `Push α down (harder text) or draft cost up and watch this go under 1×.`);
  }

  $("m7-k").oninput = () => { $("m7-k-v").textContent = K(); readout(); draw(); };
  $("m7-a").oninput = () => { $("m7-a-v").textContent = fmt(A(), 2); readout(); draw(); };
  $("m7-c").oninput = () => { $("m7-c-v").textContent = fmt(Cost() * 100, 0) + "%"; readout(); draw(); };
  $("m7-run").onclick = run;
  renderers.push(draw);
  draw(); readout();
})();

/* =========================================================
   MODULE 8 — sampling
========================================================= */
(() => {
  const canvas = $("m8-canvas");
  const TOKS = ["Paris", "the", "a", "located", "France's", "known", "Lyon", "beautiful", "in", "not", "Marseille", "banana"];
  const LOGITS = [6.2, 3.1, 2.8, 2.5, 2.2, 1.9, 1.6, 1.3, 1.1, 0.7, 0.4, -1.5];
  let counts = new Array(TOKS.length).fill(0);

  function dist() {
    const T = +$("m8-t").value, k = +$("m8-k").value, p = +$("m8-p").value;
    const scaled = LOGITS.map((l) => l / T);
    const mx = Math.max(...scaled);
    let probs = scaled.map((l) => Math.exp(l - mx));
    const Z = probs.reduce((a, b) => a + b);
    probs = probs.map((x) => x / Z);
    // rank order (LOGITS already sorted desc, but keep general)
    const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
    const keep = new Set();
    // top-k
    order.slice(0, k).forEach((i) => keep.add(i));
    // top-p intersect
    let cum = 0;
    const keepP = new Set();
    for (const i of order) { keepP.add(i); cum += probs[i]; if (cum >= p) break; }
    const final = [...keep].filter((i) => keepP.has(i));
    const Z2 = final.reduce((a, i) => a + probs[i], 0);
    const out = probs.map((q, i) => (final.includes(i) ? q / Z2 : 0));
    return { probs, out, kept: new Set(final) };
  }

  function draw() {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const { probs, out, kept } = dist();
    const pad = 40, pw = W - pad - 20, ph = H - 110;
    const bw = pw / TOKS.length;
    const pmax = Math.max(...out, ...probs, 0.15) * 1.12;
    const Y = (q) => 30 + ph - (q / pmax) * ph;

    const totalSamples = counts.reduce((a, b) => a + b, 0);
    for (let i = 0; i < TOKS.length; i++) {
      const x = pad + i * bw;
      // raw prob (ghost)
      roundRect(ctx, x + 6, Y(probs[i]), bw - 26, 30 + ph - Y(probs[i]), 3, "rgba(138,147,166,.25)");
      // filtered prob
      if (kept.has(i)) {
        roundRect(ctx, x + 12, Y(out[i]), bw - 26, 30 + ph - Y(out[i]), 3, C.teal);
      } else {
        // struck out
        ctx.strokeStyle = C.red;
        ctx.beginPath();
        ctx.moveTo(x + 8, Y(probs[i]) - 6); ctx.lineTo(x + bw - 16, 30 + ph);
        ctx.stroke();
      }
      // empirical dots
      if (totalSamples > 0 && counts[i] > 0) {
        const q = counts[i] / totalSamples;
        ctx.beginPath(); ctx.arc(x + bw / 2, Y(q), 5, 0, 7);
        ctx.fillStyle = C.amber; ctx.fill();
      }
      ctx.save();
      ctx.translate(x + bw / 2, 30 + ph + 14);
      ctx.rotate(-0.5);
      label(ctx, TOKS[i], 0, 8, kept.has(i) ? C.ink : "#555c6b", 11, "right");
      ctx.restore();
    }
    label(ctx, "grey = raw softmax   teal = after temp/top-k/top-p   ● amber = your empirical samples", pad, 18, C.dim, 11);
  }

  function readout() {
    const { out, kept } = dist();
    const H_ = -out.filter((q) => q > 0).reduce((a, q) => a + q * Math.log2(q), 0);
    $("m8-read").innerHTML =
      `candidates surviving filters: <b>${kept.size}/${TOKS.length}</b> · ` +
      `entropy of final distribution: <b>${fmt(H_, 2)} bits</b> ` +
      `(0 = deterministic/greedy, ${fmt(Math.log2(TOKS.length), 1)} = uniform) · ` +
      `samples drawn: <b>${counts.reduce((a, b) => a + b, 0)}</b>`;
  }

  function sample20() {
    const { out } = dist();
    for (let s = 0; s < 20; s++) {
      let u = Math.random(), cum = 0;
      for (let i = 0; i < out.length; i++) { cum += out[i]; if (u <= cum) { counts[i]++; break; } }
    }
    draw(); readout();
  }

  const refresh = () => { draw(); readout(); };
  $("m8-t").oninput = () => { $("m8-t-v").textContent = fmt(+$("m8-t").value, 2); refresh(); };
  $("m8-k").oninput = () => { const k = +$("m8-k").value; $("m8-k-v").textContent = k >= TOKS.length ? "off" : k; refresh(); };
  $("m8-p").oninput = () => { const p = +$("m8-p").value; $("m8-p-v").textContent = p >= 1 ? "1.00 (off)" : fmt(p, 2); refresh(); };
  $("m8-sample").onclick = sample20;
  $("m8-clear").onclick = () => { counts = counts.map(() => 0); refresh(); };
  renderers.push(refresh);
  refresh();
})();

/* =========================================================
   MODULE 9 — parallelism
========================================================= */
(() => {
  const canvas = $("m9-canvas");
  let timer = null, t = 0, events = null;

  function build() {
    const slow = $("m9-net").value === "slow";
    const comm = slow ? 0.9 : 0.25;   // all-reduce cost per layer-group
    const mb = +$("m9-mb").value;
    const NG = 4, T = 30;

    // TP: each token step = compute (0.7 split across all gpus simultaneously) + comm
    const tp = { gpus: [[], [], [], []], tokens: 0 };
    let clock = 0;
    while (clock < T) {
      for (let g = 0; g < NG; g++) tp.gpus[g].push({ t: clock, w: 0.7, kind: "comp" });
      clock += 0.7;
      for (let g = 0; g < NG; g++) tp.gpus[g].push({ t: clock, w: comm, kind: "comm" });
      clock += comm;
      if (clock <= T) tp.tokens++;
    }

    // PP: stage time 0.7 each (4 stages = same total work). microbatches pipelined.
    const pp = { gpus: [[], [], [], []], tokens: 0 };
    const stageT = 0.7, hop = 0.05;
    // simulate mb independent token streams round-robin
    const nextFree = [0, 0, 0, 0];
    const streams = [];
    for (let m = 0; m < mb; m++) streams.push({ ready: 0 });
    let guard = 0;
    while (guard++ < 400) {
      // pick stream with earliest ready
      streams.sort((a, b) => a.ready - b.ready);
      const s = streams[0];
      let tk = s.ready;
      let done = true;
      for (let g = 0; g < 4; g++) {
        const start = Math.max(tk, nextFree[g]);
        if (start + stageT > T) { done = false; break; }
        pp.gpus[g].push({ t: start, w: stageT, kind: "comp", stream: streams.indexOf(s) });
        nextFree[g] = start + stageT;
        tk = start + stageT + hop;
      }
      if (!done) break;
      s.ready = tk; // next token for this stream starts after previous exits pipeline
      pp.tokens++;
    }
    return { tp, pp, T, comm, mb };
  }

  function draw(upto) {
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    if (!events) return;
    const { tp, pp, T } = events;
    const pad = 88, pw = W - pad - 24;
    const X = (tt) => pad + (tt / T) * pw;
    const rowH = 30, gap = 6;

    const panel = (y0, name, sys, colorFn) => {
      label(ctx, name, 24, y0 - 10, C.ink, 13);
      for (let g = 0; g < 4; g++) {
        const y = y0 + g * (rowH + gap);
        label(ctx, `GPU ${g}`, pad - 10, y + rowH / 2 + 4, C.dim, 11, "right");
        ctx.fillStyle = "#0b0e14";
        ctx.fillRect(pad, y, pw, rowH);
        for (const ev of sys.gpus[g]) {
          if (ev.t > upto) continue;
          const wPix = (Math.min(ev.w, upto - ev.t) / T) * pw;
          ctx.fillStyle = colorFn(ev);
          ctx.fillRect(X(ev.t), y + 2, Math.max(wPix - 0.5, 0.5), rowH - 4);
        }
      }
    };

    panel(44, "TENSOR PARALLEL — every GPU works on every token, all-reduce after (amber)", tp,
      (ev) => (ev.kind === "comm" ? C.amber : C.teal));
    panel(216, `PIPELINE PARALLEL — layers 0-19 / 20-39 / 40-59 / 60-79, dark gaps = bubbles`, pp,
      (ev) => PALETTE[(ev.stream ?? 0) % PALETTE.length]);

    label(ctx, "time →", pad, H - 12, C.dim, 11);

    // utilization
    const util = (sys) => {
      let busy = 0;
      sys.gpus.forEach((g) => g.forEach((ev) => { if (ev.kind === "comp" && ev.t < upto) busy += Math.min(ev.w, upto - ev.t); }));
      return busy / (4 * Math.max(upto, 0.01));
    };
    label(ctx, `util ${fmt(util(tp) * 100, 0)}% · tokens ${tp.tokens}`, W - 24, 34, C.teal, 12, "right");
    label(ctx, `util ${fmt(util(pp) * 100, 0)}% · tokens ${pp.tokens}`, W - 24, 206, C.teal, 12, "right");
  }

  function readout() {
    if (!events) return;
    const { tp, pp, comm, mb, T } = events;
    const tpotTP = 0.7 + comm, tpotPP = 4 * (0.7 + 0.05);
    $("m9-read").innerHTML =
      `TP: TPOT = compute/4-GPUs + all-reduce = <b>${fmt(tpotTP, 2)}</b> units → ${tp.tokens} tokens in ${T} · ` +
      `PP: per-token latency = 4 stages = <b class="warn">${fmt(tpotPP, 2)}</b> units, but ${mb} in-flight microbatch(es) → ${pp.tokens} tokens. ` +
      `TP buys <em>latency</em> (needs fast interconnect: comm=${fmt(comm, 2)}); PP buys <em>capacity</em> and needs microbatches to hide bubbles.`;
  }

  function run() {
    clearInterval(timer);
    events = build();
    t = 0;
    timer = setInterval(() => {
      t += 0.5;
      if (t >= events.T) { t = events.T; clearInterval(timer); }
      draw(t); readout();
    }, 60);
  }

  $("m9-run").onclick = run;
  $("m9-net").onchange = () => { if (events) run(); };
  $("m9-mb").oninput = () => { $("m9-mb-v").textContent = $("m9-mb").value; if (events) run(); };
  renderers.push(() => draw(t));
  fit(canvas); // initial blank
  const { ctx, W, H } = fit(canvas);
  clear(ctx, W, H);
  label(ctx, "press Run", 30, 40, C.dim, 14);
})();

/* =========================================================
   MODULE 10 — capstone latency lab
========================================================= */
(() => {
  const canvas = $("m10-canvas");
  const GPUS = {
    a100: { name: "A100", tf: 312, bw: 2.0, mem: 80, cost: 2.5 },
    h100: { name: "H100", tf: 989, bw: 3.35, mem: 80, cost: 4.0 },
    b200: { name: "B200", tf: 2250, bw: 8.0, mem: 192, cost: 8.0 },
  };
  const MODELS = {
    8: { L: 32, kv: 8, hd: 128 },
    70: { L: 80, kv: 8, hd: 128 },
    405: { L: 126, kv: 8, hd: 128 },
  };

  function compute() {
    const P = +$("m10-model").value;                 // B params
    const gpu = GPUS[$("m10-gpu").value];
    const tp = +$("m10-tp").value;
    const wq = +$("m10-q").value;                    // bytes/param
    const kvB = +$("m10-kv").value;
    const sd = +$("m10-sd").value;
    const B = +$("m10-b").value;
    const ctxLen = +$("m10-c").value;
    const plen = +$("m10-pl").value;
    const m = MODELS[P];

    const weightGB = P * wq;
    const memTotal = gpu.mem * tp;
    const memBudget = memTotal * 0.9;
    const fits = weightGB <= memBudget;

    const kvPerTok = 2 * m.L * m.kv * m.hd * kvB;      // bytes
    const kvGB = (kvPerTok * ctxLen * B) / 1e9;
    const kvFits = weightGB + kvGB <= memBudget;
    const maxBatch = Math.max(0, Math.floor(((memBudget - weightGB) * 1e9) / (kvPerTok * ctxLen)));

    // step time (seconds): stream weights once + read whole KV of batch
    const bytesPerStep = weightGB * 1e9 + kvPerTok * ctxLen * B;
    const stepS = bytesPerStep / (tp * gpu.bw * 1e12);
    const tpotMs = (stepS * 1000) / sd;

    // TTFT: 2*P*1e9*plen FLOPs at 50% MFU across tp gpus
    const ttftMs = (2 * P * 1e9 * plen) / (tp * gpu.tf * 1e12 * 0.5) * 1000;

    const effB = Math.min(B, Math.max(maxBatch, 0));
    const tput = kvFits ? (B / (tpotMs / 1000)) : (effB > 0 ? effB / (tpotMs / 1000) : 0);
    const costPerM = tput > 0 ? (tp * gpu.cost / 3600 / tput) * 1e6 : Infinity;

    return { P, gpu, tp, wq, kvB, sd, B, ctxLen, plen, weightGB, memTotal, memBudget, fits, kvGB, kvFits, maxBatch, tpotMs, ttftMs, tput, costPerM };
  }

  function bar(ctx, x, y, w, h, frac, color, txt, target, targetFrac) {
    ctx.fillStyle = "#0b0e14"; ctx.fillRect(x, y, w, h);
    roundRect(ctx, x, y, w * clamp(frac, 0, 1), h, 4, color);
    ctx.strokeStyle = C.grid; ctx.strokeRect(x, y, w, h);
    if (targetFrac !== undefined) {
      const tx = x + w * clamp(targetFrac, 0, 1);
      ctx.strokeStyle = "#fff";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(tx, y - 4); ctx.lineTo(tx, y + h + 4); ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, target, tx, y - 8, C.dim, 10, "center");
    }
    label(ctx, txt, x + w + 12, y + h / 2 + 4, C.ink, 12);
  }

  function draw() {
    const r = compute();
    const { ctx, W, H } = fit(canvas);
    clear(ctx, W, H);
    const pad = 24, bw = W * 0.52, bh = 26, gap = 56;
    let y = 46;

    // HBM
    label(ctx, `HBM across ${r.tp}× ${r.gpu.name} (${fmt(r.memTotal, 0)} GB, 90% usable)`, pad, y - 12, C.dim, 12);
    ctx.fillStyle = "#0b0e14"; ctx.fillRect(pad, y, bw, bh);
    const wf = r.weightGB / r.memTotal, kf = r.kvGB / r.memTotal;
    roundRect(ctx, pad, y, bw * clamp(wf, 0, 1), bh, 4, C.indigo);
    roundRect(ctx, pad + bw * clamp(wf, 0, 1), y, bw * clamp(kf, 0, 1 - wf), bh, 0, r.kvFits ? C.teal : C.red);
    ctx.strokeStyle = C.grid; ctx.strokeRect(pad, y, bw, bh);
    label(ctx, `weights ${fmt(r.weightGB, 0)} GB + KV ${fmt(r.kvGB, 1)} GB ${r.kvFits ? "✓" : "✗ OOM"}`,
      pad + bw + 12, y + bh / 2 + 4, r.kvFits ? C.ink : C.red, 12);
    y += gap;

    const okC = (ok) => ok ? C.green : C.amber;
    // TPOT (target 15ms) — log-ish scale to 100ms
    bar(ctx, pad, y, bw, bh, r.tpotMs / 100, r.tpotMs <= 15 ? C.green : C.amber,
      `TPOT ${fmt(r.tpotMs, 1)} ms/token  (${fmt(1000 / r.tpotMs, 0)} tok/s per user)`, "15ms", 15 / 100);
    y += gap;
    bar(ctx, pad, y, bw, bh, r.ttftMs / 2000, r.ttftMs <= 500 ? C.green : C.amber,
      `TTFT ${fmt(r.ttftMs, 0)} ms (prefill ${fmt(r.plen, 0)} tokens)`, "500ms", 500 / 2000);
    y += gap;
    bar(ctx, pad, y, bw, bh, r.tput / 10000, r.tput >= 5000 ? C.green : C.amber,
      `throughput ${fmt(r.tput, 0)} tok/s total`, "5k", 5000 / 10000);
    y += gap;
    const costFrac = isFinite(r.costPerM) ? r.costPerM / 2 : 1;
    bar(ctx, pad, y, bw, bh, costFrac, r.costPerM <= 0.30 ? C.green : C.amber,
      `cost ${isFinite(r.costPerM) ? "$" + fmt(r.costPerM, 3) : "∞"} / 1M tokens`, "$0.30", 0.30 / 2);
    y += gap;

    label(ctx, `max batch that fits at this context: ${r.maxBatch}`, pad, y, C.dim, 12);
    if (!r.fits) label(ctx, `✗ weights alone don't fit — raise TP or quantize`, pad, y + 22, C.red, 13);

    // missions
    const hits = [r.P === 70 && r.tpotMs <= 15 && r.kvFits && r.fits,
                  r.tput >= 5000 && r.kvFits && r.fits,
                  r.costPerM <= 0.30 && r.kvFits && r.fits];
    hits.forEach((h, i) => $("ms-" + i).classList.toggle("hit", !!h));
    return r;
  }

  function readout() {
    const r = draw();
    const bound = r.kvGB > r.weightGB ? "KV-cache reads" : "weight streaming";
    $("m10-read").innerHTML =
      `per-step bytes: weights ${fmt(r.weightGB, 0)} GB + KV ${fmt(r.kvGB, 1)} GB → decode is dominated by <b>${bound}</b> · ` +
      `spec-decode multiplier ×${r.sd} applied to TPOT · ` +
      (r.kvFits ? `<span class="ok">config valid</span>` : `<span class="err">KV overflows HBM — reduce batch/context, quantize KV, or add TP</span>`);
  }

  ["m10-model", "m10-gpu", "m10-tp", "m10-q", "m10-kv", "m10-sd"].forEach((id) => $(id).onchange = readout);
  $("m10-b").oninput = () => { $("m10-b-v").textContent = $("m10-b").value; readout(); };
  $("m10-c").oninput = () => { $("m10-c-v").textContent = fmt(+$("m10-c").value, 0); readout(); };
  $("m10-pl").oninput = () => { $("m10-pl-v").textContent = fmt(+$("m10-pl").value, 0); readout(); };
  renderers.push(readout);
  readout();
})();

/* =========================================================
   MODULE 11 — exam
========================================================= */
(() => {
  const QS = [
    { q: "Decode at batch size 1 on a modern GPU is limited by…",
      opts: ["Peak FLOP/s", "HBM memory bandwidth", "PCIe bandwidth", "The Python interpreter"],
      a: 1, why: "Each token requires streaming all weights from HBM; the compute involved is tiny by comparison (arithmetic intensity ≈ 1)." },
    { q: "TTFT is dominated by which phase, and what is that phase bound by?",
      opts: ["Decode; bandwidth-bound", "Prefill; compute-bound", "Prefill; bandwidth-bound", "Sampling; CPU-bound"],
      a: 1, why: "Prefill processes the whole prompt in parallel GEMMs — lots of FLOPs, high arithmetic intensity, compute-bound." },
    { q: "The KV cache exists to…",
      opts: ["Reduce model weight memory", "Avoid recomputing K/V of past tokens every step", "Store sampled tokens", "Cache HTTP responses"],
      a: 1, why: "It trades memory capacity for compute: past tokens' keys/values are stored once and reused every subsequent step." },
    { q: "GQA reduces KV-cache size by…",
      opts: ["Quantizing keys to INT4", "Sharing each KV head across a group of query heads", "Dropping old tokens", "Compressing with gzip"],
      a: 1, why: "Fewer KV heads (e.g. 8 for 64 query heads) → 8× smaller cache and 8× less cache bandwidth per step." },
    { q: "Continuous batching improves throughput mainly by…",
      opts: ["Using bigger matrices", "Admitting/evicting requests every iteration so slots never idle", "Skipping attention layers", "Caching entire responses"],
      a: 1, why: "Iteration-level scheduling frees a slot the moment a sequence finishes instead of waiting for the whole batch." },
    { q: "PagedAttention borrows which OS concept?",
      opts: ["Spinlocks", "Virtual memory paging", "Preemptive threads", "Journaling filesystems"],
      a: 1, why: "Fixed-size KV blocks + per-sequence block tables = paging; fragmentation drops from ~60-80% to <4%." },
    { q: "Why does INT4 weight quantization speed up decode nearly 4×?",
      opts: ["INT4 ALUs are 4× faster", "4× fewer bytes stream through the memory-bound path", "It reduces layer count", "It shortens the prompt"],
      a: 1, why: "Decode time ≈ bytes/bandwidth. Quarter the bytes, quarter the time — dequantization compute is nearly free." },
    { q: "In speculative decoding, output quality vs the target model alone is…",
      opts: ["Slightly worse", "Exactly identical in distribution", "Better", "Identical only at temperature 0"],
      a: 1, why: "The accept/reject rule is constructed so the sampled distribution equals the target model's — it is lossless." },
    { q: "Which parallelism reduces per-token latency (TPOT) for a model that already fits on one GPU?",
      opts: ["Pipeline parallelism", "Tensor parallelism", "Data parallelism", "None — TPOT can't change"],
      a: 1, why: "TP splits every matrix so each GPU streams 1/N of the weights per step. PP leaves per-token latency unchanged." },
    { q: "Your server hits 100k tok/s but every request violates its latency SLO. Its goodput is…",
      opts: ["100k tok/s", "≈ 0", "50k tok/s", "Unmeasurable"],
      a: 1, why: "Goodput counts only work delivered within the SLO. Throughput without meeting deadlines is worthless for serving." },
  ];
  const exam = $("exam");
  const sel = new Array(QS.length).fill(-1);

  QS.forEach((item, qi) => {
    const div = document.createElement("div");
    div.className = "exam-q";
    div.innerHTML = `<div class="stem"><span class="n">Q${qi + 1}</span>${item.q}</div>`;
    item.opts.forEach((o, oi) => {
      const b = document.createElement("button");
      b.className = "exam-opt";
      b.textContent = o;
      b.onclick = () => {
        if (div.classList.contains("graded")) return;
        sel[qi] = oi;
        div.querySelectorAll(".exam-opt").forEach((x) => x.classList.remove("sel"));
        b.classList.add("sel");
      };
      div.appendChild(b);
    });
    const ex = document.createElement("div");
    ex.className = "exam-expl";
    ex.textContent = "→ " + item.why;
    div.appendChild(ex);
    exam.appendChild(div);
  });

  $("examGrade").onclick = () => {
    let score = 0;
    document.querySelectorAll(".exam-q").forEach((div, qi) => {
      div.classList.add("graded");
      const opts = div.querySelectorAll(".exam-opt");
      opts.forEach((b, oi) => {
        b.disabled = true;
        if (oi === QS[qi].a) b.classList.add("right");
        else if (oi === sel[qi]) b.classList.add("wrong");
      });
      if (sel[qi] === QS[qi].a) score++;
    });
    const pass = score >= 8;
    $("examResult").innerHTML = `Score: ${score}/10 — ` +
      (pass ? `<span style="color:var(--good)">PASS 🎓</span>` : `<span style="color:var(--bad)">below 8 — review and reset</span>`);
    if (pass) $("finale").hidden = false;
  };
  $("examReset").onclick = () => {
    sel.fill(-1);
    $("examResult").textContent = "";
    document.querySelectorAll(".exam-q").forEach((div) => {
      div.classList.remove("graded");
      div.querySelectorAll(".exam-opt").forEach((b) => { b.disabled = false; b.classList.remove("right", "wrong", "sel"); });
    });
  };
})();

/* =========================================================
   inline quizzes, nav, progress
========================================================= */
(() => {
  // inline quizzes
  document.querySelectorAll(".quiz-inline").forEach((qz) => {
    const fb = qz.querySelector(".qfb");
    qz.querySelectorAll(".qa").forEach((b) => {
      b.onclick = () => {
        const ok = b.dataset.ok === "1";
        qz.querySelectorAll(".qa").forEach((x) => x.classList.remove("right", "wrong"));
        b.classList.add(ok ? "right" : "wrong");
        fb.textContent = ok ? "✓ correct" : "✗ not quite — think about which resource is saturated.";
        fb.style.color = ok ? "var(--good)" : "var(--bad)";
      };
    });
  });

  // progress / completion
  const TOTAL = 12;
  const KEY = "infeng-progress";
  let done;
  try { done = new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch { done = new Set(); }

  function refresh() {
    const pct = Math.round((done.size / TOTAL) * 100);
    $("progPct").textContent = pct + "%";
    $("progFill").style.width = pct + "%";
    document.querySelectorAll(".done-btn").forEach((b) => {
      const isDone = done.has(b.dataset.m);
      b.classList.toggle("checked", isDone);
      b.textContent = isDone ? "Module complete ✓" : "Mark module complete ✓";
    });
    document.querySelectorAll("#navlist a").forEach((a) => a.classList.toggle("done", done.has(a.dataset.m)));
  }
  document.querySelectorAll(".done-btn").forEach((b) => {
    b.onclick = () => {
      done.has(b.dataset.m) ? done.delete(b.dataset.m) : done.add(b.dataset.m);
      try { localStorage.setItem(KEY, JSON.stringify([...done])); } catch {}
      refresh();
    };
  });
  refresh();

  // active section highlighting
  const links = [...document.querySelectorAll("#navlist a")];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === "#" + e.target.id));
      }
    });
  }, { rootMargin: "-30% 0px -60% 0px" });
  document.querySelectorAll(".module").forEach((m) => obs.observe(m));
})();
