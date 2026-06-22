// data.js — the "global tensor", its weight values, and how it splits into chunks.
//
// The collective operates on one big logical tensor (think: an attention weight
// matrix). We split it row-wise into `numNodes` chunks; each chunk is an
// `rows x cols` (a x b) submatrix of real-ish weight values the user can inspect.
window.CC = window.CC || {};
(function (CC) {
  CC.rows = 4; // a: rows per chunk (shard height) — tunable
  CC.cols = 4; // b: columns of the tensor          — tunable
  CC.seed = 1337;

  // Small fast deterministic PRNG (mulberry32).
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Deterministic weight value at a global (row, col) of the full tensor.
  // Real attention weights are typically Xavier/Glorot initialized — small
  // random values that mostly fall within [-1, 1]. Here we draw uniformly in
  // [-1, 1] so the numbers are easy to read. Keyed by (row,col) so a chunk's
  // submatrix and the assembled full tensor always agree.
  CC.valueAt = function (gr, c) {
    const key =
      (Math.imul(gr + 1, 73856093) ^
        Math.imul(c + 1, 19349663) ^
        Math.imul(CC.seed, 83492791)) >>> 0;
    const u = mulberry32(key)();
    return u * 2 - 1; // [-1, 1]
  };

  // Submatrix owned by `chunk` = its horizontal slice of the full tensor.
  CC.submatrix = function (chunk) {
    const m = [];
    const base = chunk * CC.rows;
    for (let r = 0; r < CC.rows; r++) {
      const row = [];
      for (let c = 0; c < CC.cols; c++) row.push(CC.valueAt(base + r, c));
      m.push(row);
    }
    return m;
  };

  // The full global tensor: N chunks stacked vertically -> (N*rows) x cols.
  // Returns an array of { chunk, values } rows so the viewer can color bands.
  CC.fullMatrix = function (numChunks) {
    const out = [];
    for (let chunk = 0; chunk < numChunks; chunk++) {
      const base = chunk * CC.rows;
      for (let r = 0; r < CC.rows; r++) {
        const row = [];
        for (let c = 0; c < CC.cols; c++) row.push(CC.valueAt(base + r, c));
        out.push({ chunk, values: row });
      }
    }
    return out;
  };

  // ---- Reduce-scatter inputs ----
  // For a reduction, every GPU contributes its OWN partial value to each chunk.
  // contribAt = GPU `rank`'s contribution to (chunk, r, c). Deterministic, keyed
  // by all four indices so contributions differ per GPU.
  CC.contribAt = function (rank, chunk, r, c) {
    const key =
      (Math.imul(rank + 1, 2654435761) ^
        Math.imul(chunk + 1, 40503) ^
        Math.imul(r + 1, 73856093) ^
        Math.imul(c + 1, 19349663) ^
        Math.imul(CC.seed, 83492791)) >>> 0;
    const u = mulberry32(key)();
    return u * 2 - 1; // [-1, 1]
  };

  CC.contribMatrix = function (rank, chunk) {
    const m = [];
    for (let r = 0; r < CC.rows; r++) {
      const row = [];
      for (let c = 0; c < CC.cols; c++) row.push(CC.contribAt(rank, chunk, r, c));
      m.push(row);
    }
    return m;
  };

  // Element-wise sum of the given ranks' contributions to `chunk` — i.e. the
  // (partial) reduction. ranks = full set [0..N-1] gives the final result.
  CC.reducedMatrix = function (chunk, ranks) {
    const m = [];
    for (let r = 0; r < CC.rows; r++) {
      const row = [];
      for (let c = 0; c < CC.cols; c++) {
        let s = 0;
        for (const rk of ranks) s += CC.contribAt(rk, chunk, r, c);
        row.push(s);
      }
      m.push(row);
    }
    return m;
  };

  // A distinct color per chunk, evenly spread around the hue wheel.
  CC.chunkColor = function (chunk, numChunks) {
    const hue = Math.round((chunk / Math.max(numChunks, 1)) * 320);
    return {
      bg: `hsl(${hue} 75% 94%)`,
      border: `hsl(${hue} 62% 58%)`,
      strong: `hsl(${hue} 58% 42%)`,
    };
  };
})(window.CC);
