// model.js — each collective as a pure state machine.
//
// A collective is a list of "steps". Each step is a full snapshot:
//   cells[i][c] : descriptor for GPU i's slot for chunk c — what the renderer draws
//     { show, fill, badge, complete, dim, data }
//       show     : render as a filled cell (vs dashed placeholder)
//       fill     : 0..1 progress (gather: present? ; reduce: contributions/N)
//       badge    : small corner text (reduce: count) or null
//       complete : fully done (all data / fully reduced) -> highlight
//       dim       : de-emphasize (a slot that will be discarded)
//       data     : what the inspector shows
//                  { kind:'plain', chunk }            -> a chunk's submatrix
//                  { kind:'reduced', chunk, ranks }   -> Σ of those ranks' contributions
//   transfers  : messages reaching this step, each { from, to, chunk }
//   title/desc : explanation
//
// The renderer is collective-agnostic: it only consumes this shape. Add a new
// collective by writing another builder that returns { name, key, numNodes, steps, cost }.
window.CC = window.CC || {};
(function (CC) {
  // ---- Ring All-Gather ----
  // GPU i starts with chunk i; each round it forwards one chunk to the right.
  // After N-1 rounds everyone holds all N chunks.
  CC.allGather = function (N) {
    const held = [];
    for (let i = 0; i < N; i++) held.push(new Set([i]));
    const steps = [];

    const snap = (transfers, title, desc) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const has = held[i].has(c);
          row.push({
            show: has,
            fill: has ? 1 : 0,
            badge: null,
            complete: has,
            dim: false,
            data: { kind: "plain", chunk: c },
          });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };

    snap([], "Step 0 · Initial",
      "Each GPU holds only its own chunk. Goal: every GPU ends up with all " + N + " chunks.");
    for (let s = 1; s <= N - 1; s++) {
      const transfers = [];
      for (let i = 0; i < N; i++) {
        const chunk = (((i - (s - 1)) % N) + N) % N;
        transfers.push({ from: i, to: (i + 1) % N, chunk });
      }
      for (const t of transfers) held[t.to].add(t.chunk);
      snap(transfers, `Step ${s} · Round ${s} of ${N - 1}`,
        `Every GPU passes a chunk to its right neighbor and receives one from its left. ` +
        `Now each GPU holds ${s + 1} of ${N} chunks.`);
    }

    return {
      name: "All-Gather", key: "all-gather", numNodes: N, steps,
      cost: {
        first: "A ring all-gather runs N−1 steps.",
        second: "Each step, every GPU sends one chunk (S bytes) to its right neighbor and receives one from its left.",
      },
    };
  };

  // ---- Ring Reduce-Scatter ----
  // Every GPU starts with its own partial contribution to ALL N chunks. Each
  // round, a GPU sends one chunk to its right neighbor, which ADDS it into its
  // matching chunk. After N-1 rounds, GPU i holds chunk i fully reduced (sum of
  // all N contributions); its other slots are partial and get discarded.
  //
  // Schedule: at round s (0-based), GPU i sends chunk (i-s-1) mod N to i+1.
  // This makes GPU i the final owner of the fully-reduced chunk i.
  CC.reduceScatter = function (N) {
    const acc = [];
    for (let i = 0; i < N; i++) {
      acc.push([]);
      for (let c = 0; c < N; c++) acc[i].push(new Set([i]));
    }
    const steps = [];

    const snap = (transfers, title, desc, isLast) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const k = acc[i][c].size;
          const owned = c === i;
          row.push({
            show: true,
            fill: k / N,
            badge: String(k),
            complete: k === N,
            dim: !!isLast && !owned,
            data: { kind: "reduced", chunk: c, ranks: [...acc[i][c]].sort((a, b) => a - b) },
          });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };

    snap([], "Step 0 · Initial",
      `Each GPU holds its own partial contribution to all ${N} chunks (1 of ${N} reduced).`, false);
    for (let s = 0; s <= N - 2; s++) {
      const sends = [];
      for (let i = 0; i < N; i++) {
        const sc = (((i - s - 1) % N) + N) % N;
        sends.push({ from: i, to: (i + 1) % N, chunk: sc, set: new Set(acc[i][sc]) });
      }
      for (const t of sends) for (const r of t.set) acc[t.to][t.chunk].add(r);
      const transfers = sends.map((t) => ({ from: t.from, to: t.to, chunk: t.chunk }));
      const isLast = s === N - 2;
      snap(transfers, `Step ${s + 1} · Round ${s + 1} of ${N - 1}`,
        isLast
          ? `Done. GPU i now holds chunk i fully reduced (sum of all ${N} contributions). Other slots are partial and discarded.`
          : `Each GPU sends one chunk to its right neighbor, which adds it into its matching chunk. Owned chunks approach full reduction.`,
        isLast);
    }

    return {
      name: "Reduce-Scatter", key: "reduce-scatter", numNodes: N, steps,
      cost: {
        first: "A ring reduce-scatter runs N−1 steps.",
        second: "Each step, every GPU sends one chunk (S bytes) to its right neighbor, which adds it into its matching chunk (reduce).",
      },
    };
  };

  CC.collectives = {
    "all-gather": { label: "All-Gather", build: CC.allGather, ready: true },
    "reduce-scatter": { label: "Reduce-Scatter", build: CC.reduceScatter, ready: true },
    "all-reduce": { label: "All-Reduce", build: null, ready: false },
    "scatter": { label: "Scatter", build: null, ready: false },
    "broadcast": { label: "Broadcast", build: null, ready: false },
    "all-to-all": { label: "All-to-All", build: null, ready: false },
  };
})(window.CC);
