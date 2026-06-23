// model.js — each collective as a pure state machine.
//
// A collective is a list of "steps". Each step is a full snapshot:
//   cells[i][c] : descriptor for GPU i's slot c — what the renderer draws
//     { show, fill, badge, complete, dim, label?, color?, data }
//       show     : render as a filled cell (vs dashed placeholder)
//       fill     : 0..1 progress (gather: present? ; reduce: contributions/N)
//       badge    : small corner text or null
//       complete : fully done -> highlight
//       dim      : de-emphasize (a slot that will be discarded)
//       label    : cell text (default "C<c>")
//       color    : index to color by (default c)
//       data     : what the inspector shows
//                  { kind:'plain', chunk }            -> a chunk's submatrix
//                  { kind:'reduced', chunk, ranks }   -> Σ of those ranks' contributions
//                  { kind:'block', src, dst }         -> an all-to-all block
//   transfers  : messages reaching this step, each
//                { from, to, chunk } or { from, to, fromChunk, toChunk, color }
//   title/desc : explanation
//   cost       : data-driven spec for the cost panel (title, rows, bullets, formula, note)
//
// The renderer is collective-agnostic: it only consumes this shape.
window.CC = window.CC || {};
(function (CC) {
  // ---------- Cost specs ----------
  const allOf = (N) => { const a = []; for (let i = 0; i < N; i++) a.push(i); return a; };

  const ROW = {
    N: { label: "GPUs <i>N</i>", value: (c) => c.N },
    S: { label: "Shard <i>S = a·b·</i>dtype", value: (c) => c.fmtBytes(c.S) },
    D: { label: "Full tensor <i>D = N·S</i>", value: (c) => c.fmtBytes(c.D) },
  };

  function ringCost(name, first, second) {
    return {
      title: name + " cost",
      rows: [
        ROW.N, ROW.S, ROW.D,
        { label: "Per-GPU sent <i>(N−1)·S</i>", value: (c) => c.fmtBytes((c.N - 1) * c.S) },
        { label: "Total, all links <i>N(N−1)·S</i>", value: (c) => c.fmtBytes(c.N * (c.N - 1) * c.S) },
        { label: "Time <i>(N−1)S ⁄ W</i>", value: (c) => c.fmtTime(((c.N - 1) * c.S) / c.W) },
      ],
      bullets: [
        first,
        second,
        "Per-GPU volume = <b>(N−1)·S = (N−1)/N · D</b> → ≈ D as N grows, so it is <b>bandwidth-optimal</b>.",
        "All N links run in parallel, so wall-clock time is set by the per-GPU volume, not the system total.",
      ],
      formula: (c) => `T = <sup>(N−1)·S</sup>&frasl;<sub>W</sub> = <span class="hl">${c.fmtTime(((c.N - 1) * c.S) / c.W)}</span>`,
      note: "Toy tensors here are tiny — the <i>formula</i> is the point. e.g. D = 1 GB, N = 8, W = 200 GB/s → T = <sup>7</sup>&frasl;<sub>8</sub>·1GB ÷ 200GB/s ≈ <b>4.4 ms</b>.",
    };
  }

  const allReduceCost = {
    title: "Ring All-Reduce cost",
    rows: [
      ROW.N, ROW.S, ROW.D,
      { label: "Per-GPU sent <i>2(N−1)·S</i>", value: (c) => c.fmtBytes(2 * (c.N - 1) * c.S) },
      { label: "Total, all links <i>2N(N−1)·S</i>", value: (c) => c.fmtBytes(2 * c.N * (c.N - 1) * c.S) },
      { label: "Time <i>2(N−1)S ⁄ W</i>", value: (c) => c.fmtTime((2 * (c.N - 1) * c.S) / c.W) },
    ],
    bullets: [
      "All-reduce = <b>reduce-scatter</b> then <b>all-gather</b>: 2(N−1) steps in two laps of the ring.",
      "Phase 1 reduces until each GPU owns one fully-reduced chunk; phase 2 shares those chunks so everyone has all.",
      "Per-GPU volume = <b>2(N−1)·S = 2(N−1)/N · D</b> → ≈ 2D as N grows.",
      "This is the bandwidth-optimal cost for all-reduce — the constant 2, not N, is what matters.",
    ],
    formula: (c) => `T = <sup>2(N−1)·S</sup>&frasl;<sub>W</sub> = <span class="hl">${c.fmtTime((2 * (c.N - 1) * c.S) / c.W)}</span>`,
    note: "Two passes over the same data — independent of N for large N.",
  };

  const scatterCost = {
    title: "Scatter cost",
    rows: [
      ROW.N, ROW.S,
      { label: "Out of root <i>(N−1)·S</i>", value: (c) => c.fmtBytes((c.N - 1) * c.S) },
      { label: "Per leaf received <i>S</i>", value: (c) => c.fmtBytes(c.S) },
      { label: "Time (root-bound) <i>(N−1)S ⁄ W</i>", value: (c) => c.fmtTime(((c.N - 1) * c.S) / c.W) },
    ],
    bullets: [
      "The root holds all N chunks and sends chunk i to GPU i.",
      "Each leaf receives exactly one shard (S); the root sends N−1 of them.",
      "If the root shares one link it is root-bound: time ≈ (N−1)·S ⁄ W. With independent links, ≈ S ⁄ W.",
    ],
    formula: (c) => `T ≈ <sup>(N−1)·S</sup>&frasl;<sub>W</sub> = <span class="hl">${c.fmtTime(((c.N - 1) * c.S) / c.W)}</span>`,
    note: "A tree/binomial scatter cuts the root bottleneck to ~log₂N hops.",
  };

  const broadcastCost = {
    title: "Broadcast cost",
    rows: [
      ROW.N, ROW.D,
      { label: "Per leaf received <i>D</i>", value: (c) => c.fmtBytes(c.D) },
      { label: "Time (linear ring) <i>(N−1)·D ⁄ W</i>", value: (c) => c.fmtTime(((c.N - 1) * c.D) / c.W) },
    ],
    bullets: [
      "The root holds the buffer; every GPU needs a full copy.",
      "This linear ring passes the whole buffer hop-by-hop: N−1 hops, each moving D.",
      "Better in practice: a pipelined ring streams chunks so large messages cost ≈ D ⁄ W; a tree costs ≈ log₂N · D ⁄ W.",
    ],
    formula: (c) => `T (linear) = <sup>(N−1)·D</sup>&frasl;<sub>W</sub> = <span class="hl">${c.fmtTime(((c.N - 1) * c.D) / c.W)}</span>`,
    note: "The visualization shows the simple linear ring for clarity.",
  };

  const allToAllCost = {
    title: "All-to-All cost",
    rows: [
      ROW.N,
      { label: "Block <i>B = a·b·</i>dtype", value: (c) => c.fmtBytes(c.S) },
      { label: "Sent per GPU <i>(N−1)·B</i>", value: (c) => c.fmtBytes((c.N - 1) * c.S) },
      { label: "Recv per GPU <i>(N−1)·B</i>", value: (c) => c.fmtBytes((c.N - 1) * c.S) },
      { label: "Time <i>(N−1)B ⁄ W</i>", value: (c) => c.fmtTime(((c.N - 1) * c.S) / c.W) },
    ],
    bullets: [
      "Each GPU sends a distinct block to every other GPU and receives one from each — a global transpose.",
      "Per GPU: N−1 blocks out and N−1 in (the diagonal block i→i stays home).",
      "With independent links all exchanges overlap, so time ≈ (N−1)·B ⁄ W.",
    ],
    formula: (c) => `T ≈ <sup>(N−1)·B</sup>&frasl;<sub>W</sub> = <span class="hl">${c.fmtTime(((c.N - 1) * c.S) / c.W)}</span>`,
    note: "All-to-all is the pattern behind tensor/sequence-parallel reshards.",
  };

  // ---------- Builders ----------

  // Ring All-Gather: GPU i starts with chunk i; forwards one chunk per round.
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
          row.push({ show: has, fill: has ? 1 : 0, badge: null, complete: has, dim: false, data: { kind: "plain", chunk: c } });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snap([], "Step 0 · Initial", "Each GPU holds only its own chunk. Goal: every GPU ends up with all " + N + " chunks.");
    for (let s = 1; s <= N - 1; s++) {
      const transfers = [];
      for (let i = 0; i < N; i++) transfers.push({ from: i, to: (i + 1) % N, chunk: (((i - (s - 1)) % N) + N) % N });
      for (const t of transfers) held[t.to].add(t.chunk);
      snap(transfers, `Step ${s} · Round ${s} of ${N - 1}`,
        `Every GPU passes a chunk to its right neighbor and receives one from its left. Now each GPU holds ${s + 1} of ${N} chunks.`);
    }
    return {
      name: "All-Gather", key: "all-gather", numNodes: N, steps,
      cost: ringCost("Ring All-Gather",
        "A ring all-gather runs N−1 steps.",
        "Each step, every GPU sends one chunk (S bytes) to its right neighbor and receives one from its left."),
    };
  };

  // Ring Reduce-Scatter: contributions accumulate around the ring; GPU i ends
  // owning chunk i fully reduced. Schedule: round s, GPU i sends chunk (i-s-1).
  CC.reduceScatter = function (N) {
    const acc = [];
    for (let i = 0; i < N; i++) { acc.push([]); for (let c = 0; c < N; c++) acc[i].push(new Set([i])); }
    const steps = [];
    const snap = (transfers, title, desc, isLast) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const k = acc[i][c].size;
          row.push({ show: true, fill: k / N, badge: String(k), complete: k === N, dim: !!isLast && c !== i,
            data: { kind: "reduced", chunk: c, ranks: [...acc[i][c]].sort((a, b) => a - b) } });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snap([], "Step 0 · Initial", `Each GPU holds its own partial contribution to all ${N} chunks (1 of ${N} reduced).`, false);
    for (let s = 0; s <= N - 2; s++) {
      const sends = [];
      for (let i = 0; i < N; i++) { const sc = (((i - s - 1) % N) + N) % N; sends.push({ from: i, to: (i + 1) % N, chunk: sc, set: new Set(acc[i][sc]) }); }
      for (const t of sends) for (const r of t.set) acc[t.to][t.chunk].add(r);
      const isLast = s === N - 2;
      snap(sends.map((t) => ({ from: t.from, to: t.to, chunk: t.chunk })), `Step ${s + 1} · Round ${s + 1} of ${N - 1}`,
        isLast
          ? `Done. GPU i now holds chunk i fully reduced (sum of all ${N} contributions). Other slots are partial and discarded.`
          : `Each GPU sends one chunk to its right neighbor, which adds it into its matching chunk. Owned chunks approach full reduction.`,
        isLast);
    }
    return {
      name: "Reduce-Scatter", key: "reduce-scatter", numNodes: N, steps,
      cost: ringCost("Ring Reduce-Scatter",
        "A ring reduce-scatter runs N−1 steps.",
        "Each step, every GPU sends one chunk (S bytes) to its right neighbor, which adds it into its matching chunk (reduce)."),
    };
  };

  // Ring All-Reduce = reduce-scatter (phase 1) + all-gather of reduced chunks (phase 2).
  CC.allReduce = function (N) {
    const steps = [];
    const allRanks = allOf(N);

    // Phase 1: reduce-scatter
    const acc = [];
    for (let i = 0; i < N; i++) { acc.push([]); for (let c = 0; c < N; c++) acc[i].push(new Set([i])); }
    const snapRS = (transfers, title, desc) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const k = acc[i][c].size;
          row.push({ show: true, fill: k / N, badge: String(k), complete: k === N, dim: false,
            data: { kind: "reduced", chunk: c, ranks: [...acc[i][c]].sort((a, b) => a - b) } });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snapRS([], "Step 0 · Initial (phase 1)", `Phase 1 — reduce-scatter. Each GPU holds its partial contribution to all ${N} chunks.`);
    for (let s = 0; s <= N - 2; s++) {
      const sends = [];
      for (let i = 0; i < N; i++) { const sc = (((i - s - 1) % N) + N) % N; sends.push({ from: i, to: (i + 1) % N, chunk: sc, set: new Set(acc[i][sc]) }); }
      for (const t of sends) for (const r of t.set) acc[t.to][t.chunk].add(r);
      snapRS(sends.map((t) => ({ from: t.from, to: t.to, chunk: t.chunk })), `Reduce-scatter ${s + 1} of ${N - 1}`,
        `Send-and-add around the ring. Owned chunks approach full reduction.`);
    }

    // Phase 2: all-gather of the fully-reduced chunks (GPU i owns chunk i).
    const held = [];
    for (let i = 0; i < N; i++) held.push(new Set([i]));
    const snapAG = (transfers, title, desc) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const has = held[i].has(c);
          row.push({ show: has, fill: has ? 1 : 0, badge: has ? String(N) : null, complete: has, dim: false,
            data: { kind: "reduced", chunk: c, ranks: allRanks } });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snapAG([], "Phase 2 begins", `Each GPU now owns one fully-reduced chunk. Discard the partial sums; all-gather the reduced shards.`);
    for (let s = 1; s <= N - 1; s++) {
      const transfers = [];
      for (let i = 0; i < N; i++) transfers.push({ from: i, to: (i + 1) % N, chunk: (((i - (s - 1)) % N) + N) % N });
      for (const t of transfers) held[t.to].add(t.chunk);
      snapAG(transfers, `All-gather ${s} of ${N - 1}`,
        `Share the reduced shards around the ring. Now each GPU holds ${s + 1} of ${N} fully-reduced chunks.`);
    }

    return { name: "All-Reduce", key: "all-reduce", numNodes: N, steps, cost: allReduceCost };
  };

  // Scatter: root (GPU 0) holds all chunks and sends chunk i to GPU i.
  CC.scatter = function (N) {
    const held = [];
    for (let i = 0; i < N; i++) held.push(new Set());
    for (let c = 0; c < N; c++) held[0].add(c);
    const steps = [];
    const snap = (transfers, title, desc) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) {
          const has = held[i].has(c);
          row.push({ show: has, fill: has ? 1 : 0, badge: null, complete: has, dim: false, data: { kind: "plain", chunk: c } });
        }
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snap([], "Step 0 · Initial", `Root GPU 0 holds all ${N} chunks; every other GPU is empty.`);
    const transfers = [];
    for (let i = 1; i < N; i++) transfers.push({ from: 0, to: i, chunk: i });
    for (const t of transfers) { held[0].delete(t.chunk); held[t.to].add(t.chunk); }
    snap(transfers, "Step 1 · Scatter", `The root sends chunk i to GPU i. Now each GPU holds only its own shard.`);
    return { name: "Scatter", key: "scatter", numNodes: N, steps, cost: scatterCost };
  };

  // Broadcast: root's buffer (all chunks) is copied to every GPU, hop by hop.
  CC.broadcast = function (N) {
    const has = [];
    for (let i = 0; i < N; i++) has.push(i === 0);
    const steps = [];
    const snap = (transfers, title, desc) => {
      const cells = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let c = 0; c < N; c++) row.push({ show: has[i], fill: has[i] ? 1 : 0, badge: null, complete: has[i], dim: false, data: { kind: "plain", chunk: c } });
        cells.push(row);
      }
      steps.push({ cells, transfers, title, desc });
    };
    snap([], "Step 0 · Initial", `Root GPU 0 holds the buffer (all ${N} chunks); the others are empty.`);
    for (let s = 1; s <= N - 1; s++) {
      const transfers = [];
      for (let c = 0; c < N; c++) transfers.push({ from: s - 1, to: s, chunk: c });
      has[s] = true;
      snap(transfers, `Step ${s} · Hop ${s} of ${N - 1}`, `GPU ${s - 1} forwards the full buffer to GPU ${s}. Now ${s + 1} GPUs have a copy.`);
    }
    return { name: "Broadcast", key: "broadcast", numNodes: N, steps, cost: broadcastCost };
  };

  // All-to-All: GPU i holds a block for every destination (i→j); after the
  // exchange GPU i holds the blocks sent to it (j→i). A global transpose.
  CC.allToAll = function (N) {
    const build = (fn) => {
      const cells = [];
      for (let i = 0; i < N; i++) { const row = []; for (let c = 0; c < N; c++) row.push(fn(i, c)); cells.push(row); }
      return cells;
    };
    const pre = (i, c) => ({ show: true, fill: 1, badge: null, complete: false, dim: false, label: `${i}→${c}`, color: i, data: { kind: "block", src: i, dst: c } });
    const post = (i, c) => ({ show: true, fill: 1, badge: null, complete: false, dim: false, label: `${c}→${i}`, color: c, data: { kind: "block", src: c, dst: i } });

    const steps = [];
    steps.push({ cells: build(pre), transfers: [], title: "Step 0 · Initial",
      desc: `GPU i holds N blocks, one per destination (label i→j). Color marks the source.` });
    const transfers = [];
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) if (a !== b) transfers.push({ from: a, to: b, fromChunk: b, toChunk: a, color: a });
    steps.push({ cells: build(post), transfers, title: "Step 1 · All-to-All",
      desc: `Every GPU sends block i→j to GPU j (a transpose). Now GPU i holds the blocks sent to it (label j→i).` });

    return { name: "All-to-All", key: "all-to-all", numNodes: N, steps, cost: allToAllCost };
  };

  CC.collectives = {
    "all-gather": { label: "All-Gather", build: CC.allGather, ready: true },
    "reduce-scatter": { label: "Reduce-Scatter", build: CC.reduceScatter, ready: true },
    "all-reduce": { label: "All-Reduce", build: CC.allReduce, ready: true },
    "scatter": { label: "Scatter", build: CC.scatter, ready: true },
    "broadcast": { label: "Broadcast", build: CC.broadcast, ready: true },
    "all-to-all": { label: "All-to-All", build: CC.allToAll, ready: true },
  };
})(window.CC);
