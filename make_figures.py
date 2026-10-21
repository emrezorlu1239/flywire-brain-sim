"""
Build every figure in the README from the data saved by ``experiments.py``.

    python make_figures.py

Outputs go to results/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from flybrain.analysis import layered_layout
from flybrain.data import load_connectome
from flybrain.neurons import MN9_LEFT, SUGAR_GRNS

DATA = Path("results/data")
FIG = Path("results/figures")

# ---- style -------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # categorical 1 / inhibitory
ORANGE = "#eb6834"   # categorical 2 / excitatory
AQUA = "#1baf7a"     # categorical 3 / MN9 highlight
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark blue ramp

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10.5, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9.5,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
    "lines.linewidth": 2,
})


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def short(fid) -> str:
    return "…" + str(fid)[-6:]


# ---- 1. validation -----------------------------------------------------------
def fig_validation():
    df = pd.read_csv(DATA / "validation.csv", index_col=0)
    inter = df[~df.sensory]
    r = np.corrcoef(inter.rate_ours, inter.rate_brian2)[0, 1]
    mn9 = df.loc[MN9_LEFT]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = max(df.rate_ours.max(), df.rate_brian2.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(inter.rate_brian2, inter.rate_ours, s=22, color=BLUE, alpha=0.75,
               edgecolor=SURFACE, linewidth=0.8, label=f"Interneurons & motor neurons ({len(inter)})", zorder=2)
    sens = df[df.sensory]
    ax.scatter(sens.rate_brian2, sens.rate_ours, s=26, color=ORANGE, edgecolor=SURFACE,
               linewidth=0.8, label=f"Stimulated sugar neurons ({len(sens)})", zorder=3)
    ax.scatter([mn9.rate_brian2], [mn9.rate_ours], s=90, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
    ax.annotate(f"MN9 (feeding motor neuron)\nours {mn9.rate_ours:.1f} Hz · Brian2 {mn9.rate_brian2:.1f} Hz",
                (mn9.rate_brian2, mn9.rate_ours), xytext=(lim * 0.04, lim * 0.8), textcoords="data",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Firing rate in published Brian2 model (Hz)")
    ax.set_ylabel("Firing rate in this PyTorch model (Hz)")
    ax.set_title("This model reproduces the published simulation", pad=22)
    subtitle(ax, f"Every active neuron, sugar neurons at 200 Hz, 30 trials · Pearson r = {r:.3f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "validation.png")
    plt.close(fig)
    return r


# ---- 2. sugar x bitter -------------------------------------------------------
def fig_sweep():
    df = pd.read_csv(DATA / "sweep_sugar_bitter.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    levels = sorted(df.bitter_hz.unique())
    for color, b in zip(ORDINAL, levels):
        d = df[df.bitter_hz == b].sort_values("sugar_hz")
        ax.fill_between(d.sugar_hz, d.mn9_hz - d.mn9_sem, d.mn9_hz + d.mn9_sem, color=color, alpha=0.18, lw=0)
        ax.plot(d.sugar_hz, d.mn9_hz, color=color, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=f"bitter {b:g} Hz")
        last = d.iloc[-1]
        ax.text(last.sugar_hz + 4, last.mn9_hz, f"bitter {b:g} Hz", color=INK_2, fontsize=9, va="center")
    ax.set_xlim(-5, 245)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sugar neuron stimulation (Hz)")
    ax.set_ylabel("MN9 firing rate (Hz)")
    ax.set_title("Bitter taste vetoes the urge to eat", pad=22)
    subtitle(ax, "Feeding motor neuron MN9 vs. sugar input, at four bitter input levels · mean ± SEM, 10 trials")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "sugar_vs_bitter.png")
    plt.close(fig)


# ---- 3. silencing --------------------------------------------------------------
def fig_silencing():
    df = pd.read_csv(DATA / "silencing.csv")
    base = df.mn9_baseline.iloc[0]
    df["change"] = 100 * (df.mn9_hz - base) / base
    df = df.sort_values("change")
    show = pd.concat([df.head(10), df.tail(4)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    y = np.arange(len(show))
    colors = [ORANGE if s > 0 else BLUE for s in show.sign]
    err = 100 * show.mn9_sem / base
    ax.barh(y, show.change, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1,
            xerr=err, error_kw=dict(ecolor=INK_2, elinewidth=0.9, capsize=0))
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, [short(f) for f in show.flywire_id], fontsize=8.5)
    for yi, (c, r) in enumerate(zip(show.change, show.rate_in_baseline)):
        e = err.iloc[yi]
        ax.text(c + (e + 1.8 if c >= 0 else -e - 1.8), yi, f"{c:+.0f}%", va="center",
                ha="left" if c >= 0 else "right", fontsize=8.5, color=INK)
    ax.invert_yaxis()
    pad = max(10, show.change.abs().max() * 0.18)
    ax.set_xlim(min(show.change.min() - pad, -pad), max(show.change.max() + pad, pad))
    ax.set_xlabel("Change in MN9 firing when this neuron is silenced (%)")
    ax.grid(axis="y", visible=False)
    ax.set_ylabel("Silenced neuron (FlyWire ID)")
    ax.set_title("One inhibitory neuron acts as a brake on feeding", pad=22)
    subtitle(ax, f"Top {len(df)} interneurons silenced one at a time · sugar 100 Hz · baseline MN9 {base:.0f} Hz · ± SEM")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="excitatory neuron"), Patch(color=BLUE, label="inhibitory neuron")],
              loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "silencing.png")
    plt.close(fig)


# ---- 4. raster + network ----------------------------------------------------------
def load_propagation():
    spikes = pd.read_parquet(DATA / "propagation_spikes.parquet")
    meta = pd.read_csv(DATA / "propagation_meta.csv", index_col=0).iloc[:, 0]
    return spikes, int(meta["trials"]), float(meta["duration_ms"])


def first_spike_latency(spikes: pd.DataFrame) -> pd.Series:
    """Median (over trials) time of each neuron's first spike."""
    return spikes.groupby(["flywire_id", "trial"]).t_ms.min().groupby("flywire_id").median()


def fig_raster(spikes, trials, duration):
    rates = spikes.groupby("flywire_id").size() / (trials * duration / 1000)
    lat = first_spike_latency(spikes)
    sugar = set(SUGAR_GRNS)
    inter = [f for f in rates.sort_values(ascending=False).index if f not in sugar][:70]
    order = list(lat.loc[[f for f in SUGAR_GRNS if f in lat.index]].sort_values().index) + \
        list(lat.loc[inter].sort_values().index)
    row = {f: i for i, f in enumerate(order)}
    tr = spikes[(spikes.trial == 0) & spikes.flywire_id.isin(row) & (spikes.t_ms <= 300)]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    is_s = tr.flywire_id.isin(sugar)
    is_m = tr.flywire_id == MN9_LEFT
    rest = ~is_s & ~is_m
    kw = dict(marker="|", s=14, linewidths=0.9)
    ax.scatter(tr.t_ms[is_s], tr.flywire_id[is_s].map(row), color=ORANGE, label="sugar neurons (stimulated)", **kw)
    ax.scatter(tr.t_ms[rest], tr.flywire_id[rest].map(row), color=BLUE, label="downstream neurons", **kw)
    ax.scatter(tr.t_ms[is_m], tr.flywire_id[is_m].map(row), color=INK, label="MN9 (feeding motor neuron)",
               marker="|", s=40, linewidths=1.6)
    ax.axhline(len([f for f in order if f in sugar]) - 0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(len(order), -1)
    ax.set_xlim(0, 300)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{len(order)} neurons, ordered by response latency")
    ax.set_title("Taste signal spreading through the brain", pad=22)
    subtitle(ax, "Single trial, first 300 ms · each tick is one spike")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(FIG / "raster.png")
    plt.close(fig)


def build_graph(c, spikes, trials, duration, n_nodes=40, min_syn=5, max_in=2, n_cols=6):
    """Sugar neurons (one node) + the top responding neurons, in columns by response latency.

    Only the strongest forward connections are drawn so the circuit stays readable.
    """
    rates = spikes.groupby("flywire_id").size() / (trials * duration / 1000)
    lat = first_spike_latency(spikes)
    sugar_set = set(SUGAR_GRNS)
    inter = [f for f in rates.sort_values(ascending=False).index if f not in sugar_set][:n_nodes]
    if MN9_LEFT not in inter and MN9_LEFT in rates.index:
        inter.append(MN9_LEFT)
    # columns: equal-count latency bins
    lat_i = lat.loc[inter].to_numpy()
    edges_q = np.quantile(lat_i, np.linspace(0, 1, n_cols + 1)[1:-1])
    col = np.concatenate([[0], 1 + np.searchsorted(edges_q, lat_i, side="right")])
    # MN9 is the read-out, so it gets its own final column on the right
    if MN9_LEFT in inter:
        col[1 + inter.index(MN9_LEFT)] = col.max() + 1
    nodes = ["sugar"] + inter
    gidx = c.index_of(inter)
    pos_of = {g: k + 1 for k, g in enumerate(gidx)}

    weights = {}
    sources = [(0, i) for i in c.index_of(SUGAR_GRNS)] + [(k + 1, g) for k, g in enumerate(gidx)]
    for a, g in sources:
        post, w = c.outgoing(g)
        for p_, w_ in zip(post, w):
            b = pos_of.get(int(p_))
            if b is not None and col[b] > col[a]:
                weights[(a, b)] = weights.get((a, b), 0.0) + float(w_)
    # For every neuron keep its strongest inputs from earlier columns:
    # "where does this neuron get its drive from?"
    edges = []
    for b in range(1, len(nodes)):
        inc = [(a, w) for (a, bb), w in weights.items() if bb == b and abs(w) >= min_syn]
        inc.sort(key=lambda x: -abs(x[1]))
        edges += [(a, b, w) for a, w in inc[:(4 if nodes[b] == MN9_LEFT else max_in)]]
    xy = layered_layout(col, edges)
    sign = np.concatenate([[1], c.sign[gidx]])
    r = np.concatenate([[rates.reindex(SUGAR_GRNS).fillna(0).mean()], rates.reindex(inter).fillna(0).to_numpy()])
    return nodes, xy, col, edges, sign, r


def draw_graph(ax, nodes, xy, edges, sign, act, vmax, show_labels=True):
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    segs, cols, lws = [], [], []
    for a, b, w in edges:
        segs.append([xy[a], xy[b]])
        alpha = 0.12 + 0.6 * min(act[a] / vmax, 1.0)
        rgb = matplotlib.colors.to_rgb(ORANGE if w > 0 else BLUE)
        cols.append((*rgb, alpha))
        lws.append(0.6 + 2.2 * min(abs(w) / 80, 1.0))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1, capstyle="round"))
    level = np.clip(act / vmax, 0, 1)
    cm_e = LinearSegmentedColormap.from_list("e", ["#f6e6de", ORANGE, "#8a2a07"])
    cm_i = LinearSegmentedColormap.from_list("i", ["#e1ebf7", BLUE, "#0d366b"])
    face = [cm_e(l) if s >= 0 else cm_i(l) for s, l in zip(sign, level)]
    sizes = 40 + 260 * level
    sizes[0] = 300 + 500 * level[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=face, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    if MN9_LEFT in nodes:
        k = nodes.index(MN9_LEFT)
        ax.scatter([xy[k, 0]], [xy[k, 1]], s=sizes[k] + 160, facecolors="none", edgecolors=INK,
                   linewidths=1.6, zorder=4)
        if show_labels:
            ax.annotate("MN9", xy[k], xytext=(14, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold", zorder=5)
    if show_labels:
        ax.annotate("21 sugar\nneurons", xy[0], xytext=(0, -24), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color=INK_2)
    ax.set_xlim(xy[:, 0].min() - 0.05, xy[:, 0].max() + 0.08)
    ax.set_ylim(xy[:, 1].min() - 0.12, xy[:, 1].max() + 0.08)


def fig_network(c, spikes, trials, duration):
    nodes, xy, col, edges, sign, rates = build_graph(c, spikes, trials, duration)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.02)
    vmax = np.percentile(rates, 95)
    draw_graph(ax, nodes, xy, edges, sign, rates, vmax)
    fig.text(0.02, 0.95, "The sugar → feeding circuit inside the connectome", fontsize=13,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.87, f"{len(nodes) - 1} most active neurons, left → right by response latency; each neuron shows its two strongest inputs\n"
             "node size & shade = firing rate · orange = excitatory, blue = inhibitory", fontsize=9.5, color=INK_2)
    fig.savefig(FIG / "network.png")
    plt.close(fig)
    return nodes, xy, edges, sign


def gif_propagation(spikes, trials, nodes, xy, edges, sign, t_max=150.0, step=2.0, smooth=6.0):
    """Animate the first `t_max` ms: trial-averaged, smoothed activity of every node."""
    pos = {f: k for k, f in enumerate(nodes)}
    pos.update({f: 0 for f in SUGAR_GRNS})          # node 0 = the 21 sugar neurons together
    sub = spikes[spikes.flywire_id.isin(pos) & (spikes.t_ms < t_max + smooth)]
    n_bins = int(np.ceil((t_max + smooth) / step))
    counts = np.zeros((n_bins, len(nodes)))
    b = (sub.t_ms // step).astype(int).clip(0, n_bins - 1)
    np.add.at(counts, (b.to_numpy(), sub.flywire_id.map(pos).to_numpy()), 1)
    rate = counts / (trials * step / 1000)
    rate[:, 0] /= len(SUGAR_GRNS)                    # mean rate per sugar neuron
    k = int(smooth / step)
    kernel = np.ones(k) / k
    rate = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="full")[:n_bins], 0, rate)
    vmax = np.percentile(rate[n_bins // 2:].max(0), 90)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03)
    title = fig.text(0.02, 0.95, "", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, "Sugar neurons switched on at t = 0 · activity spreads hop by hop to the feeding motor neuron MN9",
             fontsize=9, color=INK_2)
    frames = int(t_max / step)

    def update(i):
        draw_graph(ax, nodes, xy, edges, sign, rate[i], vmax)
        title.set_text(f"Fly brain responding to sugar   t = {i * step:5.0f} ms")
        return []

    anim = FuncAnimation(fig, update, frames=list(range(frames)) + [frames - 1] * 15, blit=False)
    anim.save(FIG / "propagation.gif", writer=PillowWriter(fps=12), dpi=100)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    done = []
    if (DATA / "validation.csv").exists():
        print("validation r =", round(fig_validation(), 4)); done.append("validation")
    if (DATA / "sweep_sugar_bitter.csv").exists():
        fig_sweep(); done.append("sweep")
    if (DATA / "silencing.csv").exists():
        fig_silencing(); done.append("silencing")
    if (DATA / "propagation_spikes.parquet").exists():
        c = load_connectome("data")
        spikes, trials, duration = load_propagation()
        fig_raster(spikes, trials, duration)
        nodes, xy, edges, sign = fig_network(c, spikes, trials, duration)
        gif_propagation(spikes, trials, nodes, xy, edges, sign)
        done.append("propagation")
    print("figures made:", ", ".join(done))


if __name__ == "__main__":
    main()
