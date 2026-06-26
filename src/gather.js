/* Indexing-ops explainer: torch.gather / torch.scatter.
   Dependency-free. Three matrices (big / index-shaped / index) plus a step
   machine that reveals one (i,j) mapping at a time and draws its arrow.

   gather : out[i][j] = input[i][index[i][j]]  (dim=1)
            out[i][j] = input[index[i][j]][j]  (dim=0)
   scatter: self[i][index[i][j]] = src[i][j]   (dim=1)
            self[index[i][j]][j] = src[i][j]   (dim=0)
*/
(function () {
  "use strict";

  const input = [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
  ];
  // Pedagogical, bounds-valid indices per dim.
  const IDX = {
    0: [[0, 1, 2, 0], [2, 2, 0, 1]], // 2×4, values in [0,2] (rows of a 3-row input)
    1: [[0, 0], [3, 1], [2, 2]],      // 3×2, values in [0,3] (cols of a 4-col input)
  };

  const cols = (m) => m[0].length;
  const srcMat = (dim) => IDX[dim].map((r, i) => r.map((_, j) => i * cols(IDX[dim]) + j + 1));
  const gather = (dim) =>
    IDX[dim].map((r, i) => r.map((v, j) => (dim === 1 ? input[i][v] : input[v][j])));
  function scatter(dim) {
    const idx = IDX[dim], src = srcMat(dim);
    const self = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    idx.forEach((r, i) => r.forEach((v, j) => {
      if (dim === 1) self[i][v] = src[i][j]; else self[v][j] = src[i][j];
    }));
    return self;
  }
  const positions = (dim) => {
    const out = [];
    IDX[dim].forEach((r, i) => r.forEach((_, j) => out.push([i, j])));
    return out;
  };
  const bigCellOf = (dim, i, j) => (dim === 1 ? [i, IDX[dim][i][j]] : [IDX[dim][i][j], j]);

  const state = { op: "gather", dim: 1, step: 0, timer: null };

  const $ = (id) => document.getElementById(id);
  const els = {
    op: $("gx-op"), dim: $("gx-dim"), step: $("gx-step"), stepVal: $("gx-step-val"),
    title: $("gx-step-title"), desc: $("gx-step-desc"), note: $("gx-note"),
    grids: $("gx-grids"), stage: $("gx-stage"), line: $("gx-line"),
    play: $("gx-play"), prev: $("gx-prev"), next: $("gx-next"), current: $("current-op"),
    insp: $("gx-inspector"), inspTitle: $("gx-inspect-title"), inspSwatch: $("gx-inspect-swatch"),
    inspBody: $("gx-inspect-body"), inspNote: $("gx-inspect-note"), inspClose: $("gx-inspect-close"),
  };

  function gridHTML(key, mat, opts) {
    // opts: { color, label, sub, cell(r,c) -> { empty, zero, active, tap } }
    let html = `<div class="gx-mat"><div class="gx-lbl">${opts.label}<small>${opts.sub}</small></div>`;
    html += `<div class="gx-grid" style="grid-template-columns:repeat(${cols(mat)},44px)">`;
    mat.forEach((row, r) => row.forEach((val, c) => {
      const s = opts.cell(r, c);
      let cls = "gx-cell ";
      if (s.empty) cls += "gx-empty";
      else if (s.zero) cls += "gx-zero";
      else cls += opts.color;
      if (s.active) cls += " gx-active";
      if (s.faint) cls += " gx-faint";
      if (s.tap) cls += " tap";
      const text = s.empty ? "" : val;
      html += `<div class="${cls}" id="${key}-${r}-${c}" data-r="${r}" data-c="${c}" data-key="${key}">${text}</div>`;
    }));
    return html + "</div></div>";
  }

  function render() {
    const dim = state.dim, idx = IDX[dim], pos = positions(dim);
    const total = pos.length, step = state.step;
    const active = step >= 1 ? pos[step - 1] : null;

    // Which big cell + index cell the active mapping touches.
    let activeBig = null;
    if (active) activeBig = bigCellOf(dim, active[0], active[1]);

    let bigMat, linkMat, bigLabel, linkLabel, bigColor, linkColor, producedIsBig;

    if (state.op === "gather") {
      bigMat = input; linkMat = gather(dim); producedIsBig = false;
      bigLabel = "input"; bigColor = "gx-blue";
      linkLabel = "output"; linkColor = "gx-teal";
    } else {
      bigMat = scatter(dim); linkMat = srcMat(dim); producedIsBig = true;
      bigLabel = "self"; bigColor = "gx-blue";
      linkLabel = "src"; linkColor = "gx-teal";
    }

    // For scatter, work out the first step at which each self cell is written.
    const firstWrite = {};
    if (producedIsBig) {
      pos.forEach((p, k) => {
        const [br, bc] = bigCellOf(dim, p[0], p[1]);
        const key = br + "," + bc;
        if (!(key in firstWrite)) firstWrite[key] = k;
      });
    }

    const isActive = (key, r, c) => {
      if (!active) return false;
      if (key === "IDX") return r === active[0] && c === active[1];
      if (key === "BIG") return activeBig && r === activeBig[0] && c === activeBig[1];
      return r === active[0] && c === active[1]; // LNK
    };

    const bigCell = (r, c) => {
      if (!producedIsBig) return { active: isActive("BIG", r, c) }; // input: always shown
      const key = r + "," + c;
      const written = key in firstWrite;
      if (!written) return { zero: true }; // never a scatter target
      const revealed = firstWrite[key] < step;
      return { empty: !revealed, active: isActive("BIG", r, c), tap: revealed };
    };

    const linkCell = (r, c) => {
      if (producedIsBig) return { active: isActive("LNK", r, c) }; // src: always shown
      const revealed = r * cols(linkMat) + c < step; // output is 1:1 with index
      return { empty: !revealed, active: isActive("LNK", r, c), tap: revealed };
    };

    els.grids.innerHTML =
      gridHTML("BIG", bigMat, { color: bigColor, label: bigLabel, sub: "3 × 4", cell: bigCell }) +
      gridHTML("LNK", linkMat, { color: linkColor, label: linkLabel, sub: idx.length + " × " + cols(idx), cell: linkCell }) +
      gridHTML("IDX", idx, {
        color: "gx-amber", label: "index", sub: idx.length + " × " + cols(idx),
        cell: (r, c) => ({ active: isActive("IDX", r, c) }),
      });

    bindTaps();
    drawArrow(active, activeBig);
    popActive(active, activeBig);
    updateText(active, total);
  }

  function drawArrow(active, activeBig) {
    if (!active) { els.line.style.display = "none"; return; }
    const [ai, aj] = active;
    const fromId = state.op === "gather" ? `BIG-${activeBig[0]}-${activeBig[1]}` : `LNK-${ai}-${aj}`;
    const toId = state.op === "gather" ? `LNK-${ai}-${aj}` : `BIG-${activeBig[0]}-${activeBig[1]}`;
    const from = $(fromId), to = $(toId);
    if (!from || !to) { els.line.style.display = "none"; return; }
    const s = els.stage.getBoundingClientRect();
    const a = from.getBoundingClientRect(), b = to.getBoundingClientRect();
    const ax = a.left + a.width / 2 - s.left, ay = a.top + a.height / 2 - s.top;
    const bx = b.left + b.width / 2 - s.left, by = b.top + b.height / 2 - s.top;
    // Stop short of the destination centre so the arrowhead sits on the cell edge.
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1, pull = 22;
    els.line.setAttribute("x1", ax + (dx / len) * pull);
    els.line.setAttribute("y1", ay + (dy / len) * pull);
    els.line.setAttribute("x2", bx - (dx / len) * pull);
    els.line.setAttribute("y2", by - (dy / len) * pull);
    els.line.style.display = "block";
  }

  function popActive(active, activeBig) {
    if (!active) return;
    const producedId = state.op === "gather"
      ? `LNK-${active[0]}-${active[1]}`
      : `BIG-${activeBig[0]}-${activeBig[1]}`;
    const el = $(producedId);
    if (el) { el.classList.remove("gx-pop"); void el.offsetWidth; el.classList.add("gx-pop"); }
  }

  function updateText(active, total) {
    const dim = state.dim;
    els.step.max = total;
    els.stepVal.textContent = state.step + " / " + total;
    els.current.textContent = state.op + " (dim=" + dim + ")";

    els.note.innerHTML = dim === 1
      ? "<strong>dim = 1</strong> → the index chooses the <strong>column</strong>; the row stays pinned to its own position. Every arrow stays inside one row."
      : "<strong>dim = 0</strong> → the index chooses the <strong>row</strong>; the column stays pinned to its own position. Every arrow stays inside one column.";

    if (!active) {
      els.title.textContent = "Start";
      els.desc.innerHTML = "Scrub <strong>Element</strong> to walk through the mapping one cell at a time.";
      return;
    }
    const [i, j] = active, v = IDX[dim][i][j];
    els.title.textContent = "Element [" + i + "][" + j + "]";
    const link = (state.op === "gather" ? gather(dim) : srcMat(dim))[i][j];
    const axis = dim === 1 ? "column" : "row";
    if (state.op === "gather") {
      // out[i][j] = input[i][index[i][j]] = input[i][v] = value   (dim=1)
      const sym = dim === 1 ? `input[${i}][index[${i}][${j}]]` : `input[index[${i}][${j}]][${j}]`;
      const resolved = dim === 1 ? `input[${i}][${v}]` : `input[${v}][${j}]`;
      els.desc.innerHTML = `<code>out[${i}][${j}] = ${sym} = ${resolved} = ${link}</code> &nbsp;— index value ${v} redirects the ${axis}.`;
    } else {
      // self[i][index[i][j]] = self[i][v] = src[i][j] = value      (dim=1)
      const sym = dim === 1 ? `self[${i}][index[${i}][${j}]]` : `self[index[${i}][${j}]][${j}]`;
      const resolved = dim === 1 ? `self[${i}][${v}]` : `self[${v}][${j}]`;
      els.desc.innerHTML = `<code>${sym} = ${resolved} = src[${i}][${j}] = ${link}</code> &nbsp;— src element ${link} is written to the ${axis} index ${v} names.`;
    }
  }

  /* ---- Inspector: provenance of a produced cell ---- */
  function bindTaps() {
    els.grids.querySelectorAll(".gx-cell.tap").forEach((el) =>
      el.addEventListener("click", () => inspect(+el.dataset.r, +el.dataset.c, el.dataset.key)));
  }

  function inspect(r, c, key) {
    const dim = state.dim;
    if (state.op === "gather" && key === "LNK") {
      const v = IDX[dim][r][c];
      const [br, bc] = bigCellOf(dim, r, c);
      els.inspSwatch.style.background = "#e1f5ee";
      els.inspTitle.textContent = `output[${r}][${c}]`;
      els.inspBody.innerHTML = rows([
        ["index value", `index[${r}][${c}] = ${v}`],
        ["reads from", `input[${br}][${bc}]`],
        ["value", gather(dim)[r][c]],
      ]);
      els.inspNote.innerHTML = dim === 1
        ? `Row <b>${r}</b> is pinned (output row = input row). Only the column moved: ${c} → ${v}.`
        : `Column <b>${c}</b> is pinned (output col = input col). Only the row moved: ${r} → ${v}.`;
    } else if (state.op === "scatter" && key === "BIG") {
      // Find which index entry wrote this self cell (last writer wins).
      const writers = [];
      positions(dim).forEach(([i, j]) => {
        const [br, bc] = bigCellOf(dim, i, j);
        if (br === r && bc === c) writers.push([i, j]);
      });
      const last = writers[writers.length - 1];
      els.inspSwatch.style.background = "#e6f1fb";
      els.inspTitle.textContent = `self[${r}][${c}]`;
      els.inspBody.innerHTML = rows([
        ["written by", `src[${last[0]}][${last[1]}]`],
        ["because", `index[${last[0]}][${last[1]}] = ${IDX[dim][last[0]][last[1]]}`],
        ["value", scatter(dim)[r][c]],
      ]);
      els.inspNote.innerHTML = writers.length > 1
        ? `<b>${writers.length} source cells</b> target this slot — last write wins. <code>scatter_add</code> would sum them instead.`
        : "Exactly one source cell targets this slot.";
    }
    els.insp.classList.add("open");
  }

  const rows = (pairs) =>
    `<table class="matrix" style="width:100%"><tbody>` +
    pairs.map(([k, val]) =>
      `<tr><td style="text-align:left;border:none;color:var(--muted)">${k}</td>` +
      `<td style="border:none;font-weight:700">${val}</td></tr>`).join("") +
    `</tbody></table>`;

  /* ---- Real-world example: cross-entropy loss via gather ---- */
  // log_softmax scores [T, V]; each row's gold token sits near 0, others lower.
  const LP = [
    [-2.1, -0.3, -3.0, -2.5, -1.8],
    [-2.8, -2.2, -0.4, -1.9, -2.0],
    [-1.5, -2.6, -2.9, -0.5, -2.1],
    [-2.0, -2.4, -1.7, -2.8, -0.6],
  ];
  const TGT = [1, 2, 3, 4]; // gold next-token id per position: cat, sat, on, mat

  function renderExample() {
    const host = $("gx-ex-grids");
    if (!host) return;
    const lpDisp = LP.map((row) => row.map((v) => v.toFixed(1)));
    const tgtCol = TGT.map((t) => [t]);
    const nllCol = TGT.map((t, r) => [LP[r][t].toFixed(1)]);
    host.innerHTML =
      gridHTML("EXLP", lpDisp, {
        color: "gx-blue", label: "log-probs", sub: "T × V · position × vocab",
        cell: (r, c) => ({ active: c === TGT[r], faint: c !== TGT[r] }),
      }) +
      gridHTML("EXT", tgtCol, {
        color: "gx-amber", label: "target id", sub: "gold next token",
        cell: () => ({}),
      }) +
      gridHTML("EXN", nllCol, {
        color: "gx-teal", label: "gathered", sub: "logprob of gold token",
        cell: () => ({ active: true }),
      });
    const picks = TGT.map((t, r) => LP[r][t]);
    const mean = picks.reduce((a, v) => a - v, 0) / picks.length;
    $("gx-ex-loss").innerHTML =
      `<code>loss = −mean( logprobs.gather(-1, targets) ) = −mean(${picks.map((v) => v.toFixed(1)).join(", ")}) = ${mean.toFixed(2)}</code>`;
  }

  /* ---- Controls ---- */
  function setStep(n) {
    const total = positions(state.dim).length;
    state.step = Math.max(0, Math.min(total, n));
    els.step.value = state.step;
    render();
  }
  function reset() {
    stop();
    els.step.max = positions(state.dim).length;
    setStep(0);
  }
  function stop() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; els.play.textContent = "▶ Play"; }
  }
  function play() {
    const total = positions(state.dim).length;
    if (state.timer) { stop(); return; }
    if (state.step >= total) setStep(0);
    els.play.textContent = "⏸ Pause";
    state.timer = setInterval(() => {
      if (state.step >= total) { stop(); return; }
      setStep(state.step + 1);
    }, 850);
  }

  els.op.addEventListener("change", (e) => { state.op = e.target.value; reset(); });
  els.dim.addEventListener("change", (e) => { state.dim = +e.target.value; reset(); });
  els.step.addEventListener("input", (e) => { stop(); setStep(+e.target.value); });
  els.prev.addEventListener("click", () => { stop(); setStep(state.step - 1); });
  els.next.addEventListener("click", () => { stop(); setStep(state.step + 1); });
  els.play.addEventListener("click", play);
  els.inspClose.addEventListener("click", () => els.insp.classList.remove("open"));
  window.addEventListener("resize", () => render());

  // Sync state to whatever the controls actually show (browsers may restore a
  // previously-selected <select> value across reloads).
  state.op = els.op.value;
  state.dim = +els.dim.value;
  reset();
  renderExample();
})();
