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
  let inspectMode = null; // { type:'chunk', chunk } | { type:'full' } | null

  function rebuild() {
    const N = parseInt($("node-count").value, 10);
    $("node-count-val").textContent = N;

    model = CC.allGather(N);
    CC.Render.build(els, model, (chunk) => openInspector(chunk));

    const stepSlider = $("step");
    stepSlider.max = String(model.steps.length - 1);
    step = 0;
    stepSlider.value = "0";
    updateStepLabel();
    updateCostModel();
    CC.Render.renderStep(0, false);
    // a full-tensor view stays valid across N changes; a chunk view may not
    if (inspectMode && inspectMode.type === "chunk" && inspectMode.chunk >= N) {
      closeInspector();
    } else if (inspectMode) {
      renderInspector();
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

  // ---- Cost model (ring all-gather) ----
  function fmtBytes(n) {
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1000 && i < u.length - 1) {
      n /= 1000;
      i++;
    }
    const v = i === 0 ? n : n.toFixed(n < 10 ? 2 : 1);
    return v + " " + u[i];
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

    const S = CC.rows * CC.cols * bytes; // shard bytes
    const D = N * S; // full tensor bytes
    const perGpu = (N - 1) * S; // each GPU sends this much
    const total = N * (N - 1) * S; // summed across all links
    const T = perGpu / W; // seconds (links run in parallel)

    $("cm-n").textContent = N;
    $("cm-s").textContent = fmtBytes(S);
    $("cm-d").textContent = fmtBytes(D);
    $("cm-pergpu").textContent = fmtBytes(perGpu);
    $("cm-total").textContent = fmtBytes(total);
    $("cm-time").textContent = fmtTime(T);
  }

  // ---- Inspector ----
  const fmt = (v) => (v < 0 ? "" : " ") + v.toFixed(2); // align sign column

  function openInspector(chunk) {
    inspectMode = { type: "chunk", chunk };
    renderInspector();
  }
  function openFullTensor() {
    inspectMode = { type: "full" };
    renderInspector();
  }

  function renderInspector() {
    if (!inspectMode) return;
    const N = model.numNodes;

    if (inspectMode.type === "chunk") {
      const chunk = inspectMode.chunk;
      const color = CC.chunkColor(chunk, N);
      const m = CC.submatrix(chunk);

      $("inspect-title").textContent = "Chunk C" + chunk;
      const sw = $("inspect-swatch");
      sw.style.background = color.bg;
      sw.style.borderColor = color.border;

      let html = "<table class='matrix'><tbody>";
      for (const row of m) {
        html += "<tr>";
        for (const v of row) html += `<td>${fmt(v)}</td>`;
        html += "</tr>";
      }
      html += "</tbody></table>";
      $("inspect-matrix").innerHTML = html;

      const base = chunk * CC.rows;
      $("inspect-note").textContent =
        `${CC.rows}×${CC.cols} shard · rows ${base}–${base + CC.rows - 1} of the ` +
        `full ${N * CC.rows}×${CC.cols} tensor. Values: Glorot-style weights in [-1, 1].`;
    } else {
      $("inspect-title").textContent = "Full tensor";
      const sw = $("inspect-swatch");
      sw.style.background = "#eef1f6";
      sw.style.borderColor = "#cdd2dc";

      const fm = CC.fullMatrix(N);
      let html = "<table class='matrix full'><tbody>";
      let prevChunk = -1;
      for (let i = 0; i < fm.length; i++) {
        const { chunk, values } = fm[i];
        const color = CC.chunkColor(chunk, N);
        html += `<tr style="background:${color.bg}">`;
        if (chunk !== prevChunk) {
          html +=
            `<td class="band" rowspan="${CC.rows}" ` +
            `style="color:${color.strong};border-color:${color.border}">C${chunk}</td>`;
          prevChunk = chunk;
        }
        for (const v of values) html += `<td>${fmt(v)}</td>`;
        html += "</tr>";
      }
      html += "</tbody></table>";
      $("inspect-matrix").innerHTML = html;

      $("inspect-note").textContent =
        `Full tensor = ${N} chunks × ${CC.rows} rows = ${N * CC.rows}×${CC.cols}. ` +
        `Each color band is one chunk; in all-gather every GPU ends up holding all of them.`;
    }

    $("inspector").classList.add("open");
  }

  function closeInspector() {
    inspectMode = null;
    $("inspector").classList.remove("open");
  }

  // ---- Events ----
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
  $("toggle-explain").addEventListener("click", () => {
    const hidden = $("sidebar").classList.toggle("hidden");
    const btn = $("toggle-explain");
    btn.textContent = hidden ? "▸ Show cost model" : "▾ Hide cost model";
    btn.setAttribute("aria-expanded", String(!hidden));
  });
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

  let resizeRAF = null;
  window.addEventListener("resize", () => {
    if (resizeRAF) cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(() => CC.Render.refresh(step));
  });

  rebuild();
})(window.CC);
