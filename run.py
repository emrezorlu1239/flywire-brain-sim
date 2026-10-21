"""
Playground: run one whole-brain simulation from config.yaml and report what happened.

    python run.py
    python run.py --set stimulate.sugar=150 --set stimulate.bitter=100
    python run.py --config my_config.yaml

Prints the read-out neurons (MN9 by default), the most active neurons and how far the
activity spread, then saves the rates (CSV) and a raster plot to output.save_dir.
"""

from __future__ import annotations

import argparse
import difflib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from flybrain.data import load_connectome
from flybrain.model import BrainSimulator, LIFParams, Stimulus
from flybrain.neurons import GROUPS, READOUTS


ALLOWED = {
    "simulation": {"duration_ms", "n_trials", "seed", "device"},
    "stimulate": None,   # group names, checked in build_stimuli
    "silence": None,
    "readout": None,
    "neuron": {f.name for f in fields(LIFParams)} - {"duration_ms", "n_trials"},
    "output": {"top_n", "save_dir", "plot"},
}


def validate(cfg: dict) -> None:
    """Fail loudly on misspelled keys instead of silently ignoring them."""
    def check(name, allowed, where):
        if name not in allowed:
            hint = difflib.get_close_matches(name, allowed, n=1)
            tip = f" Did you mean '{hint[0]}'?" if hint else f" Allowed: {sorted(allowed)}"
            raise SystemExit(f"Unknown config key '{where}{name}'.{tip}")

    for section, value in cfg.items():
        check(section, ALLOWED, "")
        if ALLOWED[section] and isinstance(value, dict):
            for key in value:
                check(key, ALLOWED[section], f"{section}.")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(value)
    return cfg


def build_stimuli(c, cfg) -> list[Stimulus]:
    stims = []
    for name, spec in (cfg.get("stimulate") or {}).items():
        if isinstance(spec, dict):
            ids, hz = spec["ids"], spec["hz"]
        else:
            if name not in GROUPS:
                raise KeyError(f"Unknown group '{name}'. Known: {list(GROUPS)}")
            ids, hz = GROUPS[name], spec
        if hz and hz > 0:
            stims.append(Stimulus(name, c.index_of(ids), float(hz)))
    return stims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. stimulate.sugar=150")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    validate(cfg)

    sim_cfg = cfg["simulation"]
    device = sim_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    params = LIFParams(**{k: float(v) for k, v in (cfg.get("neuron") or {}).items()},
                       duration_ms=float(sim_cfg["duration_ms"]), n_trials=int(sim_cfg["n_trials"]))

    print("Loading connectome ...")
    c = load_connectome("data")
    print(f"  {c.n_neurons:,} neurons, {c.n_connections:,} connections")

    stims = build_stimuli(c, cfg)
    silence = c.index_of(cfg.get("silence") or [])
    readout = {name: READOUTS.get(name, name) for name in cfg.get("readout") or []}
    print("Stimulus: " + (", ".join(f"{s.name} {s.rate_hz:g} Hz" for s in stims) or "none"))
    if silence:
        print(f"Silenced: {len(silence)} neuron(s)")

    sim = BrainSimulator(c, params, device=device, seed=sim_cfg.get("seed"))
    print(f"Simulating {params.n_trials} trial(s) x {params.duration_ms:g} ms on {device} ...")
    res = sim.run(stims, silence=silence, verbose=True)
    rates = res.rates()

    stim_idx = {i for s in stims for i in s.indices}
    active = np.flatnonzero(rates > 0)
    print(f"\nDone in {res.wall_time_s:.1f}s. {len(active)} neurons fired "
          f"({len(active) - len(stim_idx & set(active.tolist()))} of them not directly stimulated).")

    print("\nRead-out neurons:")
    for name, fid in readout.items():
        i = c.index_of([fid])[0]
        per_trial = res.rates_per_trial([i])[:, 0]
        if len(per_trial) > 1:
            sem = per_trial.std(ddof=1) / np.sqrt(len(per_trial))
            spread = f"(± {sem:.1f} SEM)"
        else:
            spread = "(SEM n/a: run more than 1 trial)"
        print(f"  {name:<12} {per_trial.mean():7.1f} Hz  {spread}")

    top_n = cfg.get("output", {}).get("top_n", 20)
    silenced = set(silence)
    order = [i for i in np.argsort(-rates) if rates[i] > 0 and i not in stim_idx][:top_n]
    table = pd.DataFrame({
        "flywire_id": c.flywire_ids[order],
        "rate_hz": rates[order].round(1),
        "type": ["excitatory" if c.sign[i] > 0 else "inhibitory" for i in order],
        "note": ["SILENCED" if i in silenced else "" for i in order],
    })
    print(f"\nTop {len(order)} responding (non-stimulated) neurons:")
    print(table.to_string(index=False))
    if silenced & set(order):
        print("  SILENCED = its outgoing synapses are cut. It still receives input and fires,\n"
              "  but its spikes reach no other neuron (like optogenetic silencing of release).")

    out = Path(cfg.get("output", {}).get("save_dir", "results/playground"))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"flywire_id": c.flywire_ids[active], "rate_hz": rates[active],
                  "stimulated": [i in stim_idx for i in active]}) \
        .sort_values("rate_hz", ascending=False).to_csv(out / "rates.csv", index=False)
    print(f"\nAll rates saved to {out / 'rates.csv'}")

    if cfg.get("output", {}).get("plot", True):
        plot_raster(c, res, order[:40], readout, out / "raster.png", stims)
        print(f"Raster plot saved to {out / 'raster.png'}")


def plot_raster(c, res, order, readout, path, stims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [c.index_of([f])[0] for f in readout.values()]
    rows = list(dict.fromkeys(list(order) + ids))
    pos = {n: k for k, n in enumerate(rows)}
    sel = (res.trials == 0) & np.isin(res.neurons, rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.array([pos[n] for n in res.neurons[sel]])
    is_ro = np.isin(res.neurons[sel], ids)
    ax.scatter(res.times_ms[sel][~is_ro], y[~is_ro], marker="|", s=12, color="#2a78d6", linewidths=0.8)
    ax.scatter(res.times_ms[sel][is_ro], y[is_ro], marker="|", s=30, color="#0b0b0b", linewidths=1.5)
    ax.set_yticks([pos[i] for i in ids], list(readout))
    ax.set_ylim(len(rows), -1)
    ax.set_xlim(0, res.params.duration_ms)
    ax.set_xlabel("Time (ms)")
    title = ", ".join(f"{s.name} {s.rate_hz:g} Hz" for s in stims) or "no stimulus"
    ax.set_title(f"Trial 1 - top responding neurons + read-outs ({title})", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
