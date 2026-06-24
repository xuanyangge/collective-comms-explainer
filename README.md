# Collective Communication, visually

An interactive, dependency-free explainer for the collective communication
operations behind distributed deep-learning training. The goal is something you
can *play with* to build intuition — tune the number of GPUs, scrub through the
algorithm step by step, inspect the actual data each GPU holds, and see the
data-volume / time cost model.

Implemented operations (switch via the Operation dropdown): **All-Gather**,
**Reduce-Scatter**, **All-Reduce** (ring), plus **Scatter**, **Broadcast**, and
**All-to-All**. They all share one collective-agnostic renderer — each algorithm
is just a pure state machine in `src/model.js`.

## Features

- **Tunable ring size** — 2–8 GPUs.
- **Step scrubber + play/pause** — watch chunks propagate one neighbor per round.
- **Tappable submatrices** — each chunk is a real `a × b` weight submatrix
  (Glorot-style values in `[-1, 1]`); tap any block to inspect its numbers. In
  reduce-scatter, tapping shows the **running partial sum** and which GPUs'
  contributions have been reduced in so far (a `k/N` badge per slot).
- **Tunable chunk shape** — adjust rows `a` and cols `b` live.
- **Full-tensor view** — assemble all chunks into the complete global tensor.
- **Cost model panel** (collapsible, at the bottom) — live data volume and time:
  - shard `S = a·b·dtype`, full tensor `D = N·S`
  - per-GPU bytes sent `(N−1)·S = (N−1)/N · D`
  - wall-clock time `T = (N−1)·S / W` for per-device bandwidth `W`
  - derivation of why ring all-gather is **bandwidth-optimal**.

## Run it

It's plain HTML + SVG + JavaScript with **no build step**.

- Easiest: open `index.html` directly in a browser.
- Or serve the folder (avoids any future module/CORS issues):

  ```sh
  python -m http.server 5500
  # then visit http://localhost:5500
  ```

## Project structure

| File            | Role |
|-----------------|------|
| `index.html`    | Layout: controls, stage, inspector, cost panel |
| `styles.css`    | Styling + pop/fly animations |
| `src/data.js`   | Global tensor → per-chunk submatrices, weights, colors |
| `src/model.js`  | Collective algorithms as pure step-by-step state machines |
| `src/render.js` | Node cards, tappable chunk cells, arrows, flying blocks |
| `src/main.js`   | Wires controls, inspector, and the cost model |

To add a new collective, write a builder in `src/model.js` that returns the same
`{ name, numNodes, steps[] }` shape — the renderer needs no changes.

## Roadmap

- [x] All-Gather
- [x] Reduce-Scatter
- [x] All-Reduce (reduce-scatter + all-gather; cost `2·(N−1)/N·D`)
- [x] Scatter
- [x] Broadcast
- [x] All-to-All (bidirectional-ring / TPU torus, ⌊N/2⌋ steps)
- [ ] Optional latency (α) term in the cost model
- [ ] Tree / pipelined variants for broadcast & scatter
