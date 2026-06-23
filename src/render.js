// render.js — draws collective snapshots into the DOM. Collective-agnostic:
// it only reads the generic cell descriptors produced by model.js.
window.CC = window.CC || {};
(function (CC) {
  const Render = {};

  let els = {};
  let model = null;
  let onInspect = null; // callback(node, chunk)

  Render.build = function (refs, m, inspectCb) {
    els = refs;
    model = m;
    onInspect = inspectCb;

    const N = model.numNodes;
    const grid = els.nodesGrid;
    grid.innerHTML = "";

    for (let i = 0; i < N; i++) {
      const card = document.createElement("div");
      card.className = "node";
      card.dataset.node = String(i);

      const title = document.createElement("div");
      title.className = "node-title";
      title.textContent = "GPU " + i;
      card.appendChild(title);

      const cells = document.createElement("div");
      cells.className = "node-cells";

      for (let c = 0; c < N; c++) {
        const color = CC.chunkColor(c, N);
        const cell = document.createElement("button");
        cell.className = "cell";
        cell.type = "button";
        cell.dataset.node = String(i);
        cell.dataset.chunk = String(c);
        cell.style.setProperty("--bg", color.bg);
        cell.style.setProperty("--border", color.border);
        cell.style.setProperty("--strong", color.strong);
        cell.innerHTML =
          '<span class="cell-fill"></span>' +
          '<span class="cell-label">C' + c + "</span>" +
          '<span class="cell-badge"></span>';
        cell.addEventListener("click", () => {
          if (cell.classList.contains("empty")) return; // nothing to inspect
          if (onInspect) onInspect(i, c);
        });
        cells.appendChild(cell);
      }

      card.appendChild(cells);
      grid.appendChild(card);
    }
  };

  function cellEl(node, chunk) {
    return els.nodesGrid.querySelector(
      `.cell[data-node="${node}"][data-chunk="${chunk}"]`
    );
  }

  function relRect(el) {
    const a = el.getBoundingClientRect();
    const b = els.stage.getBoundingClientRect();
    return {
      x: a.left - b.left,
      y: a.top - b.top,
      w: a.width,
      h: a.height,
      cx: a.left - b.left + a.width / 2,
      cy: a.top - b.top + a.height / 2,
    };
  }

  Render.renderStep = function (stepIndex, animate) {
    const step = model.steps[stepIndex];
    const prev = stepIndex > 0 ? model.steps[stepIndex - 1] : null;
    const N = model.numNodes;

    for (let i = 0; i < N; i++) {
      for (let c = 0; c < N; c++) {
        const d = step.cells[i][c];
        const cell = cellEl(i, c);
        cell.classList.toggle("empty", !d.show);
        cell.classList.toggle("filled", d.show);
        cell.classList.toggle("complete", !!d.complete);
        cell.classList.toggle("dim", !!d.dim);
        cell.querySelector(".cell-fill").style.height =
          (d.show ? Math.round(d.fill * 100) : 0) + "%";
        cell.querySelector(".cell-badge").textContent = d.badge || "";
        cell.querySelector(".cell-label").textContent = d.label || "C" + c;

        const color = CC.chunkColor(d.color == null ? c : d.color, N);
        cell.style.setProperty("--bg", color.bg);
        cell.style.setProperty("--border", color.border);
        cell.style.setProperty("--strong", color.strong);

        cell.classList.remove("just-arrived");
        if (prev) {
          const p = prev.cells[i][c];
          const grew = (d.show && !p.show) || d.fill > p.fill + 1e-9;
          if (grew) {
            void cell.offsetWidth; // restart the pop animation
            cell.classList.add("just-arrived");
          }
        }
      }
    }

    clearOverlay();
    if (step.transfers.length) {
      drawArrows(step.transfers);
      if (animate) flyTransfers(step.transfers);
    }

    els.stepTitle.textContent = step.title;
    els.stepDesc.textContent = step.desc;
  };

  function clearOverlay() {
    [...els.overlay.querySelectorAll(".arrow")].forEach((n) => n.remove());
    [...els.stage.querySelectorAll(".fly")].forEach((n) => n.remove());
  }

  function drawArrows(transfers) {
    const NS = "http://www.w3.org/2000/svg";
    for (const t of transfers) {
      const from = relRect(document.querySelector(`.node[data-node="${t.from}"]`));
      const to = relRect(document.querySelector(`.node[data-node="${t.to}"]`));
      const path = document.createElementNS(NS, "path");
      const x1 = from.cx, y1 = from.y - 6;
      const x2 = to.cx, y2 = to.y - 6;
      const midY = Math.min(y1, y2) - 26 - Math.abs(t.to - t.from) * 6;
      path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
      path.setAttribute("class", "arrow");
      path.setAttribute("marker-end", "url(#arrowhead)");
      els.overlay.appendChild(path);
    }
  }

  function flyTransfers(transfers) {
    for (const t of transfers) {
      const fc = t.fromChunk == null ? t.chunk : t.fromChunk;
      const tc = t.toChunk == null ? t.chunk : t.toChunk;
      const colorIdx = t.color == null ? fc : t.color;
      const src = cellEl(t.from, fc);
      const dst = cellEl(t.to, tc);
      if (!src || !dst) continue;
      const a = relRect(src);
      const b = relRect(dst);
      const color = CC.chunkColor(colorIdx, model.numNodes);

      const fly = document.createElement("div");
      fly.className = "fly";
      fly.textContent = src.querySelector(".cell-label").textContent;
      fly.style.left = a.x + "px";
      fly.style.top = a.y + "px";
      fly.style.width = a.w + "px";
      fly.style.height = a.h + "px";
      fly.style.setProperty("--bg", color.bg);
      fly.style.setProperty("--border", color.border);
      fly.style.setProperty("--strong", color.strong);
      els.stage.appendChild(fly);

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const anim = fly.animate(
        [
          { transform: "translate(0px,0px) scale(1)", opacity: 0.95 },
          { transform: `translate(${dx * 0.5}px, ${dy - 34}px) scale(1.08)`, opacity: 1, offset: 0.5 },
          { transform: `translate(${dx}px, ${dy}px) scale(1)`, opacity: 0.2 },
        ],
        { duration: 720, easing: "cubic-bezier(.45,.05,.3,1)" }
      );
      const remove = () => fly.remove();
      anim.onfinish = remove;
      setTimeout(remove, 900); // fallback cleanup
    }
  }

  Render.refresh = function (stepIndex) {
    Render.renderStep(stepIndex, false);
  };

  CC.Render = Render;
})(window.CC);
