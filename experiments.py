"""
Reproduce every result and figure in the README.

    python experiments.py            # run all experiments (~25 min on a 2-core CPU)
    python experiments.py --only sweep silencing
    python make_figures.py           # turn the saved data into figures

Experiments
-----------
1. validation   Sugar neurons at 200 Hz. Compare every neuron's firing rate with the
                published Brian2 simulation from Shiu et al. (2024).
2. sweep        Sugar x bitter: how does the feeding motor neuron MN9 respond when sweet
                and bitter taste are mixed? (bitter should suppress feeding)
3. silencing    Silence each of the most active interneurons one at a time. Which ones
                does the sugar -> MN9 pathway actually depend on?
4. propagation  A long, many-trial run kept at full spike resolution for the raster plot,
                the network graph and the propagation animation.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flybrain.data import _fetch, load_connectome
from flybrain.model import BrainSimulator, LIFParams, Stimulus
from flybrain.neurons import BITTER_GRNS, MN9_LEFT, SUGAR_GRNS

OUT = Path("results/data")
DEVICE = "cpu"
REF_URL = ("https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
           "results/example/sugarR.parquet")


def mn9_stats(res, mn9):
    r = res.rates_per_trial([mn9])[:, 0]
    return r.mean(), r.std(ddof=1) / np.sqrt(len(r))


def validation(c):
    ref_path = Path("data/brian2_reference_sugar200.parquet")
    if not ref_path.exists():
        _fetch(REF_URL, ref_path)
    ref = pd.read_parquet(ref_path)
    n_ref = ref["trial"].nunique()
    theirs = (ref.groupby("flywire_id").size() / n_ref).reindex(c.flywire_ids).fillna(0)

    p = LIFParams(n_trials=30)
    res = BrainSimulator(c, p, device=DEVICE, seed=1).run([Stimulus("sugar", c.index_of(SUGAR_GRNS), 200)])
    ours = pd.Series(res.rates(), index=c.flywire_ids)
    df = pd.DataFrame({"rate_ours": ours, "rate_brian2": theirs})
    df = df[(df.rate_ours > 0) | (df.rate_brian2 > 0)]
    df["sensory"] = df.index.isin(SUGAR_GRNS)
    df.index.name = "flywire_id"
    df.to_csv(OUT / "validation.csv")
    print(f"validation: {len(df)} active neurons, wall {res.wall_time_s:.0f}s")


def sweep(c, trials=10):
    sim = BrainSimulator(c, LIFParams(n_trials=trials), device=DEVICE, seed=2)
    mn9 = c.index_of([MN9_LEFT])[0]
    sugar, bitter = c.index_of(SUGAR_GRNS), c.index_of(BITTER_GRNS)
    rows = []
    for b_hz in [0, 50, 100, 200]:
        for s_hz in range(0, 201, 25):
            res = sim.run([Stimulus("sugar", sugar, s_hz), Stimulus("bitter", bitter, b_hz)])
            m, se = mn9_stats(res, mn9)
            rows.append(dict(sugar_hz=s_hz, bitter_hz=b_hz, mn9_hz=m, mn9_sem=se))
            print(f"sweep sugar {s_hz:3d} Hz  bitter {b_hz:3d} Hz  ->  MN9 {m:6.1f} Hz")
            pd.DataFrame(rows).to_csv(OUT / "sweep_sugar_bitter.csv", index=False)


def silencing(c, n_candidates=30, trials=8, sugar_hz=100):
    sim = BrainSimulator(c, LIFParams(n_trials=trials), device=DEVICE, seed=3)
    mn9 = c.index_of([MN9_LEFT])[0]
    sugar = c.index_of(SUGAR_GRNS)
    stim = [Stimulus("sugar", sugar, sugar_hz)]
    base = sim.run(stim)
    rates = base.rates()
    base_m, base_se = mn9_stats(base, mn9)

    order = np.argsort(-rates)
    skip = set(sugar) | {mn9}
    cands = [int(i) for i in order if rates[i] > 0 and int(i) not in skip][:n_candidates]
    rows = []
    for k, i in enumerate(cands):
        res = sim.run(stim, silence=[i])
        m, se = mn9_stats(res, mn9)
        rows.append(dict(flywire_id=int(c.flywire_ids[i]), rate_in_baseline=rates[i],
                         sign=int(c.sign[i]), mn9_hz=m, mn9_sem=se,
                         mn9_baseline=base_m, mn9_baseline_sem=base_se))
        print(f"silencing {k + 1:2d}/{len(cands)} {c.flywire_ids[i]} -> MN9 {m:5.1f} "
              f"(baseline {base_m:.1f})")
        pd.DataFrame(rows).to_csv(OUT / "silencing.csv", index=False)


def propagation(c, trials=30, duration_ms=500.0, sugar_hz=150):
    p = LIFParams(n_trials=trials, duration_ms=duration_ms)
    res = BrainSimulator(c, p, device=DEVICE, seed=4).run([Stimulus("sugar", c.index_of(SUGAR_GRNS), sugar_hz)])
    df = pd.DataFrame({"t_ms": res.times_ms.astype(np.float32), "trial": res.trials.astype(np.int16),
                       "flywire_id": c.flywire_ids[res.neurons]})
    df.to_parquet(OUT / "propagation_spikes.parquet")
    meta = pd.Series({"trials": trials, "duration_ms": duration_ms, "sugar_hz": sugar_hz})
    meta.to_csv(OUT / "propagation_meta.csv")
    print(f"propagation: {len(df)} spikes from {df.flywire_id.nunique()} neurons")


EXPERIMENTS = {"validation": validation, "sweep": sweep,
               "silencing": silencing, "propagation": propagation}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    DEVICE = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    conn = load_connectome("data")
    for name in args.only:
        EXPERIMENTS[name](conn)
