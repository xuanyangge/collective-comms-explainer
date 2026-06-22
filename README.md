# Collective Communication, visually

An interactive, dependency-free explainer for the collective communication
operations behind distributed deep-learning training. The goal is something you
can *play with* to build intuition — tune the number of GPUs, scrub through the
algorithm step by step, inspect the actual data each GPU holds, and see the
data-volume / time cost model.

Currently implemented: **Ring All-Gather**. The architecture is built so the
other collectives (all-reduce, reduce-scatter, scatter, broadcast, all-to-all)
plug into the same renderer.

## Features

- **Tunable ring size** — 2–8 GPUs.
- **Step scrubber + play/pause** — watch chunks propagate one neighbor per round.
- **Tappable submatrices** — each chunk is a real `a × b` weight submatrix
  (Glorot-style values in `[-1, 1]`); tap any block to inspect its numbers.
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

- [ ] All-Reduce (reduce-scatter + all-gather; cost `2·(N−1)/N·D`)
- [ ] Reduce-Scatter
- [ ] Scatter / Broadcast
- [ ] All-to-All
- [ ] Optional latency (α) term in the cost model
