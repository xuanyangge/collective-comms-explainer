// model.js — the algorithm as a pure state machine.
//
// A collective is described as a list of "steps". Each step is a full snapshot:
//   held[i]    : Set of chunk indices that GPU i holds at this step
//   transfers  : the messages that happened to REACH this step (from prev step)
//                each = { from, to, chunk }
//   title/desc : human-readable explanation for the step
//
// The renderer never needs to know the algorithm — it just draws snapshots.
// To add a new collective (all-reduce, scatter, ...), add another builder here
// that returns the same shape.
window.CC = window.CC || {};
(function (CC) {
  function snapshot(held) {
    return held.map((s) => new Set(s));
  }

  // Ring all-gather.
  // Start: GPU i holds only chunk i.
  // Each round: every GPU sends one chunk to its right neighbor (i -> i+1) and
  // receives one from its left. After N-1 rounds, everyone holds all N chunks.
  CC.allGather = function (numNodes) {
    const N = numNodes;
    const held = [];
    for (let i = 0; i < N; i++) held.push(new Set([i]));

    const steps = [
      {
        held: snapshot(held),
        transfers: [],
        title: "Step 0 · Initial",
        desc: "Each GPU holds only its own chunk. Goal: every GPU ends up with all " + N + " chunks.",
      },
    ];

    for (let s = 1; s <= N - 1; s++) {
      const transfers = [];
      for (let i = 0; i < N; i++) {
        // The chunk GPU i forwards this round is the one it received last round
        // (its own chunk in round 1). Index walks backward around the ring.
        const chunk = (((i - (s - 1)) % N) + N) % N;
        transfers.push({ from: i, to: (i + 1) % N, chunk });
      }
      for (const t of transfers) held[t.to].add(t.chunk);

      steps.push({
        held: snapshot(held),
        transfers,
        title: `Step ${s} · Round ${s} of ${N - 1}`,
        desc:
          `Every GPU passes a chunk to its right neighbor and receives one from its left. ` +
          `Now each GPU holds ${s + 1} of ${N} chunks.`,
      });
    }

    return { name: "All-Gather", numNodes: N, steps };
  };

  // Registry so the UI can list available collectives. Others are stubs for now.
  CC.collectives = {
    "all-gather": { label: "All-Gather", build: CC.allGather, ready: true },
    "all-reduce": { label: "All-Reduce", build: null, ready: false },
    "reduce-scatter": { label: "Reduce-Scatter", build: null, ready: false },
    "scatter": { label: "Scatter", build: null, ready: false },
    "broadcast": { label: "Broadcast", build: null, ready: false },
    "all-to-all": { label: "All-to-All", build: null, ready: false },
  };
})(window.CC);
