// render.js — draws algorithm snapshots into the DOM.
//
// Layout: a row of node "cards" (GPUs). Each card holds N stacked cells, one per
// chunk index. A cell is "filled" (colored) if that GPU holds the chunk, else a
// dashed placeholder. Filled cells are tappable -> opens the inspector with the
// chunk's submatrix. A transparent SVG overlay on top draws the transfer arrows,
// and short-lived "flying" blocks animate data moving between cards.
window.CC = window.CC || {};
(function (CC) {
  const Render = {};

  let els = {};        // cached DOM references
  let model = null;    // current collective model
  let onInspect = null; // callback(chunk) when a cell is tapped

  // Build the node cards + cells for a fresh model. Called when N or collective
  // changes. Returns nothing; subsequent renderStep() calls just toggle state.
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

      for (let chunk = 0; chunk < N; chunk++) {
        const color = CC.chunkColor(chunk, N);
        const cell = document.createElement("button");
        cell.className = "cell empty";
        cell.type = "button";
        cell.dataset.node = String(i);
        cell.dataset.chunk = String(chunk);
        cell.style.setProperty("--bg", color.bg);
        cell.style.setProperty("--border", color.border);
        cell.style.setProperty("--strong", color.strong);
        cell.textContent = "C" + chunk;
        cell.addEventListener("click", () => {
          if (cell.classList.contains("empty")) return; // nothing to inspect
          if (onInspect) onInspect(chunk);
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

  // Rectangle of an element relative to the stage (the positioned overlay root).
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

  // Render a snapshot. animate=true plays the flying blocks + arrows for the
  // transfers that produced this step.
  Render.renderStep = function (stepIndex, animate) {
    const step = model.steps[stepIndex];
    const prev = stepIndex > 0 ? model.steps[stepIndex - 1] : null;
    const N = model.numNodes;

    // Toggle every cell to match the snapshot; mark freshly-arrived ones.
    for (let i = 0; i < N; i++) {
      for (let chunk = 0; chunk < N; chunk++) {
        const cell = cellEl(i, chunk);
        const has = step.held[i].has(chunk);
        const hadBefore = prev ? prev.held[i].has(chunk) : has;
        cell.classList.toggle("empty", !has);
        cell.classList.toggle("filled", has);
        cell.classList.remove("just-arrived");
        if (has && !hadBefore) {
          // restart the pop animation
          void cell.offsetWidth;
          cell.classList.add("just-arrived");
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
    const svg = els.overlay;
    // keep <defs>, drop drawn paths
    [...svg.querySelectorAll(".arrow")].forEach((n) => n.remove());
    [...els.stage.querySelectorAll(".fly")].forEach((n) => n.remove());
  }

  function drawArrows(transfers) {
    const svg = els.overlay;
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
      svg.appendChild(path);
    }
  }

  // Decorative motion: a colored block flies from sender's chunk cell to the
  // receiver's chunk cell. Purely cosmetic — the snapshot is already correct.
  function flyTransfers(transfers) {
    for (const t of transfers) {
      const src = cellEl(t.from, t.chunk);
      const dst = cellEl(t.to, t.chunk);
      if (!src || !dst) continue;
      const a = relRect(src);
      const b = relRect(dst);
      const color = CC.chunkColor(t.chunk, model.numNodes);

      const fly = document.createElement("div");
      fly.className = "fly";
      fly.textContent = "C" + t.chunk;
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
      // Fallback: ensure cleanup even if onfinish never fires.
      setTimeout(remove, 900);
    }
  }

  // Reposition overlay drawings on resize (cards may reflow).
  Render.refresh = function (stepIndex) {
    Render.renderStep(stepIndex, false);
  };

  CC.Render = Render;
})(window.CC);
