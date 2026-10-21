"""
Graph helpers used for the network figures.

* ``hop_layers``: how many synapses away from the stimulated neurons is each neuron?
  (breadth-first search over the connectome, ignoring weak connections)
* ``layered_layout``: place neurons in columns by hop distance, and order each column
  so that connected neurons sit near each other (fewer crossing lines).
"""

from __future__ import annotations

import numpy as np

from .data import Connectome


def hop_layers(c: Connectome, sources: list[int], targets: list[int], min_syn: int = 5) -> np.ndarray:
    """Shortest synaptic path length from any source to each target (sources -> 0)."""
    indptr, post, w = c.indptr.numpy(), c.post.numpy(), c.weight.numpy()
    dist = np.full(c.n_neurons, -1)
    dist[sources] = 0
    frontier = np.asarray(sources)
    want = set(int(t) for t in targets)
    d = 0
    while len(frontier) and any(dist[t] < 0 for t in want):
        d += 1
        starts, ends = indptr[frontier], indptr[frontier + 1]
        syn = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)]) if len(frontier) else []
        syn = syn[np.abs(w[syn]) >= min_syn]
        nxt = np.unique(post[syn])
        nxt = nxt[dist[nxt] < 0]
        dist[nxt] = d
        frontier = nxt
        if d > 12:
            break
    out = dist[np.asarray(targets)]
    out[out < 0] = out.max() + 1
    return out


def layered_layout(layer: np.ndarray, edges: list[tuple[int, int, float]], sweeps: int = 4) -> np.ndarray:
    """(n, 2) positions: x from layer, y ordered by the barycenter of neighbours."""
    n = len(layer)
    layers = sorted(set(layer.tolist()))
    x_of = {l: i / max(len(layers) - 1, 1) for i, l in enumerate(layers)}
    y = np.zeros(n)
    members = {l: np.where(layer == l)[0] for l in layers}
    for l in layers:  # initial even spread
        m = members[l]
        y[m] = np.linspace(0, 1, len(m)) if len(m) > 1 else 0.5

    nbrs = [[] for _ in range(n)]
    for a, b, _ in edges:
        if a != b:
            nbrs[a].append(b)
            nbrs[b].append(a)

    for _ in range(sweeps):
        for l in layers:
            m = members[l]
            bary = np.array([np.mean(y[nbrs[i]]) if nbrs[i] else y[i] for i in m])
            order = m[np.argsort(bary, kind="stable")]
            k = len(order)
            # big columns use the full height, small ones stay centred
            span = min(1.0, 0.12 * (k - 1)) if k > 1 else 0
            y[order] = 0.5 - span / 2 + (np.arange(k) / max(k - 1, 1)) * span

    # stagger crowded columns into two sub-columns so dots don't overlap
    x = np.array([x_of[l] for l in layer], dtype=float)
    gap = 1 / max(len(layers) - 1, 1)
    for l in layers:
        m = members[l]
        if len(m) > 14:
            order = m[np.argsort(y[m])]
            x[order[1::2]] += gap * 0.28
    return np.column_stack([x, y])
