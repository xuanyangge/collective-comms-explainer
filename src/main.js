// main.js — wires the controls to the model + renderer.
window.CC = window.CC || {};
(function (CC) {
  const $ = (id) => document.getElementById(id);

  const els = {
    stage: $("stage"),
    nodesGrid: $("nodes-grid"),
    overlay: $("overlay"),
    stepTitle: $("step-title"),
    stepDesc: $("step-desc"),
  };

  let model = null;
  let step = 0;
  let playing = false;
  let playTimer = null;
  let inspectMode = null; // { type:'cell', node, chunk } | { type:'full' } | null

  const OP_NOTES = {
    "all-gather": "Each message carries one <strong>full chunk</strong>; a slot is either a complete copy or empty.",
    "reduce-scatter":
      "Each message still carries a <strong>full chunk</strong> — the entire a×b submatrix. " +
      "The fill bar and <strong>k/N</strong> badge show <strong>reduction progress</strong> " +
      "(how many GPUs' contributions are summed in), not a partial amount of data in flight.",
    "all-reduce":
      "Phase 1 reduces (the <strong>k/N</strong> badge counts contributions summed in); phase 2 shares the finished shards. " +
      "Every message carries a full chunk.",
    "scatter": "The root splits its tensor and sends shard i to GPU i. Each message is one <strong>full chunk</strong>.",
    "broadcast": "The root's buffer is copied to every GPU, chunk by chunk. Each message carries a <strong>full chunk</strong>.",
    "all-to-all":
      "Labels read <strong>letter+number = destination+source</strong> (A=GPU0, B=GPU1, …), colored by destination. " +
      "Each GPU starts with one block per destination and ends holding every block addressed to it (all one letter). " +
      "Bidirectional ring: nearest partners exchange first, then distance-2, … (⌊N/2⌋ steps).",
  };

  function currentBuilder() {
    const c = CC.collectives[$("collective").value];
    return c && c.ready && c.build ? c.build : CC.allGather;
  }

  function rebuild() {
    const N = parseInt($("node-count").value, 10);
    $("node-count-val").textContent = N;

    model = currentBuilder()(N);
    CC.Render.build(els, model, openInspector);
    const ring = ["all-gather", "reduce-scatter", "all-reduce"].includes(model.key);
    $("current-op").textContent = (ring ? "Ring " : "") + model.name;
    $("op-note").innerHTML = OP_NOTES[model.key] || "";
    $("full-tensor").style.display = model.key === "all-to-all" ? "none" : "";

    const stepSlider = $("step");
    stepSlider.max = String(model.steps.length - 1);
    step = 0;
    stepSlider.value = "0";
    updateStepLabel();
    updateCostModel();
    CC.Render.renderStep(0, false);

    if (inspectMode) {
      if (inspectMode.type === "cell" && (inspectMode.node >= N || inspectMode.chunk >= N)) {
        closeInspector();
      } else {
        renderInspector();
      }
    }
  }

  function updateStepLabel() {
    $("step-val").textContent = step + " / " + (model.steps.length - 1);
  }

  function goToStep(next, animate) {
    const clamped = Math.max(0, Math.min(model.steps.length - 1, next));
    $("step").value = String(clamped);
    if (clamped === step) return; // no-op: don't replay the animation
    step = clamped;
    updateStepLabel();
    CC.Render.renderStep(step, animate);
    if (inspectMode && inspectMode.type === "cell") renderInspector();
  }

  function play() {
    if (playing) return;
    if (step >= model.steps.length - 1) goToStep(0, false);
    playing = true;
    $("play").textContent = "⏸ Pause";
    const tick = () => {
      if (!playing) return;
      if (step >= model.steps.length - 1) {
        stop();
        return;
      }
      goToStep(step + 1, true);
      playTimer = setTimeout(tick, 1050);
    };
    playTimer = setTimeout(tick, 150);
  }

  function stop() {
    playing = false;
    $("play").textContent = "▶ Play";
    if (playTimer) clearTimeout(playTimer);
  }

  // ---- Cost model (ring all-gather / reduce-scatter — same communication cost) ----
  function fmtBytes(n) {
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1000 && i < u.length - 1) {
      n /= 1000;
      i++;
    }
    return (i === 0 ? n : n.toFixed(n < 10 ? 2 : 1)) + " " + u[i];
  }
  function fmtTime(s) {
    if (s < 1e-6) return (s * 1e9).toFixed(2) + " ns";
    if (s < 1e-3) return (s * 1e6).toFixed(2) + " µs";
    if (s < 1) return (s * 1e3).toFixed(2) + " ms";
    return s.toFixed(2) + " s";
  }
  function updateCostModel() {
    const N = model.numNodes;
    const bytes = parseInt($("dtype").value, 10);
    const W = Math.max(1, parseFloat($("bw").value) || 1) * 1e9; // per-device GB/s -> bytes/s
    const S = CC.rows * CC.cols * bytes;
    const D = N * S;
    const ctx = { N, S, D, W, fmtBytes, fmtTime };
    const cost = model.cost;

    $("cm-title").innerHTML = cost.title;
    $("cm-rows").innerHTML = cost.rows
      .map((r) => `<div class="side-row"><span>${r.label}</span><b>${r.value(ctx)}</b></div>`)
      .join("");
    $("cm-derive").innerHTML = cost.bullets.map((b) => `<li>${b}</li>`).join("");
    $("cm-formula").innerHTML = typeof cost.formula === "function" ? cost.formula(ctx) : cost.formula;
    $("cm-note").innerHTML = cost.note || "";
  }

  // ---- Inspector ----
  const fmt = (v) => (v < 0 ? "" : " ") + v.toFixed(2);

  function renderMatrix(m) {
    let html = "<table class='matrix'><tbody>";
    for (const row of m) {
      html += "<tr>";
      for (const v of row) html += `<td>${fmt(v)}</td>`;
      html += "</tr>";
    }
    html += "</tbody></table>";
    $("inspect-matrix").innerHTML = html;
  }

  function openInspector(node, chunk) {
    inspectMode = { type: "cell", node, chunk };
    renderInspector();
  }
  function openFullTensor() {
    inspectMode = { type: "full" };
    renderInspector();
  }

  function renderInspector() {
    if (!inspectMode) return;
    const N = model.numNodes;
    const sw = $("inspect-swatch");

    if (inspectMode.type === "cell") {
      const { node, chunk } = inspectMode;
      const d = model.steps[step].cells[node][chunk];
      const color = CC.chunkColor(d.color == null ? chunk : d.color, N);
      sw.style.background = color.bg;
      sw.style.borderColor = color.border;

      if (d.data.kind === "plain") {
        $("inspect-title").textContent = "Chunk C" + chunk;
        renderMatrix(CC.submatrix(chunk));
        const base = chunk * CC.rows;
        $("inspect-note").textContent =
          `${CC.rows}×${CC.cols} shard · rows ${base}–${base + CC.rows - 1} of the ` +
          `full ${N * CC.rows}×${CC.cols} tensor. Glorot-style weights in [-1, 1].`;
      } else if (d.data.kind === "reduced") {
        const ranks = d.data.ranks;
        $("inspect-title").textContent = `GPU ${node} · Chunk C${chunk}`;
        renderMatrix(CC.reducedMatrix(chunk, ranks));
        $("inspect-note").textContent =
          `Full ${CC.rows}×${CC.cols} submatrix · ${ranks.length}/${N} reduced — ` +
          `element-wise Σ of contributions from GPU ${ranks.join(", ")}. ` +
          (ranks.length === N
            ? `Fully reduced ✓ — GPU ${chunk} owns this shard.`
            : `Partial sum (still accumulating).`);
      } else {
        const { src, dst } = d.data;
        $("inspect-title").textContent = `Block ${CC.letter(dst)}${src}`;
        renderMatrix(CC.blockMatrix(src, dst));
        $("inspect-note").textContent =
          `Full ${CC.rows}×${CC.cols} block · from GPU ${src}, destined for GPU ${dst} (letter ${CC.letter(dst)}). ` +
          (src === dst ? `Stays home on GPU ${dst}.` : (dst === node ? `Arrived home on GPU ${node}.` : `Currently on GPU ${node}, still to be sent.`));
      }
    } else {
      const reduce = model.key === "reduce-scatter" || model.key === "all-reduce";
      $("inspect-title").textContent = reduce ? "Reduced result" : "Full tensor";
      sw.style.background = "#eef1f6";
      sw.style.borderColor = "#cdd2dc";

      const allRanks = [];
      for (let i = 0; i < N; i++) allRanks.push(i);

      let html = "<table class='matrix full'><tbody>";
      for (let chunk = 0; chunk < N; chunk++) {
        const color = CC.chunkColor(chunk, N);
        const m = reduce ? CC.reducedMatrix(chunk, allRanks) : CC.submatrix(chunk);
        for (let r = 0; r < CC.rows; r++) {
          html += `<tr style="background:${color.bg}">`;
          if (r === 0) {
            html +=
              `<td class="band" rowspan="${CC.rows}" ` +
              `style="color:${color.strong};border-color:${color.border}">C${chunk}</td>`;
          }
          for (let c = 0; c < CC.cols; c++) html += `<td>${fmt(m[r][c])}</td>`;
          html += "</tr>";
        }
      }
      html += "</tbody></table>";
      $("inspect-matrix").innerHTML = html;

      $("inspect-note").textContent = reduce
        ? `Fully-reduced result = element-wise Σ over all ${N} GPUs, per chunk (${N * CC.rows}×${CC.cols}).`
        : `Full tensor = ${N} chunks × ${CC.rows} rows = ${N * CC.rows}×${CC.cols}.`;
    }

    $("inspector").classList.add("open");
  }

  function closeInspector() {
    inspectMode = null;
    $("inspector").classList.remove("open");
  }

  // ---- Events ----
  $("collective").addEventListener("change", () => {
    stop();
    rebuild();
  });
  $("node-count").addEventListener("input", () => {
    stop();
    rebuild();
  });
  $("rows").addEventListener("input", (e) => {
    CC.rows = parseInt(e.target.value, 10);
    $("rows-val").textContent = CC.rows;
    updateCostModel();
    if (inspectMode) renderInspector();
  });
  $("cols").addEventListener("input", (e) => {
    CC.cols = parseInt(e.target.value, 10);
    $("cols-val").textContent = CC.cols;
    updateCostModel();
    if (inspectMode) renderInspector();
  });
  $("dtype").addEventListener("change", updateCostModel);
  $("bw").addEventListener("input", updateCostModel);
  $("full-tensor").addEventListener("click", openFullTensor);
  $("step").addEventListener("input", (e) => {
    stop();
    goToStep(parseInt(e.target.value, 10), true);
  });
  $("prev").addEventListener("click", () => {
    stop();
    goToStep(step - 1, true);
  });
  $("next").addEventListener("click", () => {
    stop();
    goToStep(step + 1, true);
  });
  $("play").addEventListener("click", () => (playing ? stop() : play()));
  $("inspect-close").addEventListener("click", closeInspector);
  $("toggle-explain").addEventListener("click", () => {
    const hidden = $("sidebar").classList.toggle("hidden");
    const btn = $("toggle-explain");
    btn.textContent = hidden ? "▸ Show cost model" : "▾ Hide cost model";
    btn.setAttribute("aria-expanded", String(!hidden));
  });

  let resizeRAF = null;
  window.addEventListener("resize", () => {
    if (resizeRAF) cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(() => CC.Render.refresh(step));
  });

  rebuild();
})(window.CC);
