"""
Leaky integrate-and-fire (LIF) simulation of the whole fruit fly brain.

Every one of the ~127,000 neurons follows the same two equations (Shiu et al. 2024):

    dv/dt = (v_rest - v + g) / tau_m      membrane voltage leaks back to rest
    dg/dt = -g / tau_syn                  synaptic input decays exponentially

* When ``v`` crosses ``v_threshold`` the neuron SPIKES: ``v`` and ``g`` are reset and the
  neuron is silent for a short refractory period.
* Every spike reaches each postsynaptic partner after ``delay_ms`` and changes its ``g`` by
  ``weight_mV x (signed synapse count)``: positive for excitatory, negative for inhibitory.
* "Stimulating" a neuron means driving it with random (Poisson) spikes at a given rate,
  which mimics optogenetic activation in a real fly. "Silencing" a neuron removes all
  of its outgoing synapses.

Only a handful of parameters exist and none of them were fitted to behaviour - all the
"intelligence" comes from the wiring diagram itself. That is the point of the project.

Implementation notes
--------------------
* Several trials run in parallel as a batch dimension: state tensors are (trials, neurons).
* Between spikes the equations are linear, so we integrate them *exactly* over each time
  step (no Euler error), the same thing Brian2's ``method='linear'`` does.
* Synaptic delay is a ring buffer of future inputs. When neurons spike we gather only
  their outgoing synapses from the CSR matrix and scatter-add them into the buffer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import Connectome


@dataclass
class LIFParams:
    """Model constants. Defaults are the published values (units: mV, ms, Hz)."""

    v_rest: float = -52.0        # resting potential (Kakaria & de Bivort 2017)
    v_reset: float = -52.0       # voltage right after a spike
    v_threshold: float = -45.0   # spike threshold
    tau_m: float = 20.0          # membrane time constant
    tau_syn: float = 5.0         # synaptic time constant (Jurgensen et al. 2021)
    refractory_ms: float = 2.2   # silent period after a spike (Lazar et al. 2021)
    delay_ms: float = 1.8        # synaptic transmission delay (Paul et al. 2015)
    weight_mV: float = 0.275     # voltage per synapse - the only free parameter
    poisson_weight_mV: float = 68.75  # kick from one stimulus event (0.275 x 250): always spikes
    dt: float = 0.1              # integration step
    duration_ms: float = 1000.0  # length of one trial
    n_trials: int = 10           # independent random trials, run in parallel


@dataclass
class Stimulus:
    """A group of neurons driven with Poisson spikes at ``rate_hz``."""

    name: str
    indices: list[int]
    rate_hz: float


@dataclass
class SimResult:
    """Spikes of one simulation. Each spike is (time step, trial, neuron index)."""

    steps: np.ndarray
    trials: np.ndarray
    neurons: np.ndarray
    params: LIFParams
    n_neurons: int
    wall_time_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def times_ms(self) -> np.ndarray:
        return self.steps * self.params.dt

    def rates(self) -> np.ndarray:
        """Mean firing rate (Hz) of every neuron, averaged over trials."""
        counts = np.bincount(self.neurons, minlength=self.n_neurons)
        return counts / (self.params.n_trials * self.params.duration_ms / 1000.0)

    def rates_per_trial(self, idx: list[int] | np.ndarray) -> np.ndarray:
        """(trials, len(idx)) firing rates for selected neurons - used for error bars."""
        idx = np.asarray(idx)
        out = np.zeros((self.params.n_trials, len(idx)))
        pos = {int(n): k for k, n in enumerate(idx)}
        mask = np.isin(self.neurons, idx)
        for t, n in zip(self.trials[mask], self.neurons[mask]):
            out[t, pos[int(n)]] += 1
        return out / (self.params.duration_ms / 1000.0)

    def binned_rates(self, idx, bin_ms: float = 10.0) -> np.ndarray:
        """(time bins, len(idx)) trial-averaged rates over time - used for animations."""
        idx = np.asarray(idx)
        n_bins = int(math.ceil(self.params.duration_ms / bin_ms))
        pos = np.full(self.n_neurons, -1)
        pos[idx] = np.arange(len(idx))
        sel = pos[self.neurons] >= 0
        b = np.minimum((self.times_ms[sel] // bin_ms).astype(int), n_bins - 1)
        out = np.zeros((n_bins, len(idx)))
        np.add.at(out, (b, pos[self.neurons[sel]]), 1)
        return out / (self.params.n_trials * bin_ms / 1000.0)


class BrainSimulator:
    def __init__(self, connectome: Connectome, params: LIFParams | None = None,
                 device: str = "cpu", seed: int | None = 0):
        self.c = connectome
        self.p = params or LIFParams()
        self.device = torch.device(device)
        self.seed = seed
        self.indptr = connectome.indptr.to(self.device)
        self.post = connectome.post.to(self.device)
        self.base_weight = (connectome.weight * self.p.weight_mV).to(self.device)

    @torch.no_grad()
    def run(self, stimuli: list[Stimulus], silence: list[int] | None = None,
            verbose: bool = False) -> SimResult:
        p, dev = self.p, self.device
        B, N = p.n_trials, self.c.n_neurons
        n_steps = int(round(p.duration_ms / p.dt))
        delay_steps = int(round(p.delay_ms / p.dt))
        ref_steps = int(round(p.refractory_ms / p.dt))
        gen = torch.Generator(device=dev)
        if self.seed is not None:
            gen.manual_seed(self.seed)

        # Silencing = delete outgoing synapses of the chosen neurons.
        weight = self.base_weight
        if silence:
            weight = weight.clone()
            for i in silence:
                weight[self.indptr[i]:self.indptr[i + 1]] = 0.0

        # Exact one-step solution of the linear ODEs.
        a_v = math.exp(-p.dt / p.tau_m)
        a_g = math.exp(-p.dt / p.tau_syn)
        c_g = p.tau_syn / (p.tau_syn - p.tau_m) * (a_g - a_v)

        # --- "Awake" neurons -------------------------------------------------------
        # A neuron that has never received any input sits exactly at rest (v = v_rest,
        # g = 0) and stays there, so integrating it is wasted work. We keep state only
        # for neurons that are stimulated or have been targeted by at least one spike,
        # and wake new neurons up the moment a spike is sent to them. Results are
        # identical to updating all ~127k neurons, just several times faster.
        local = torch.full((N,), -1, dtype=torch.long, device=dev)  # global -> state column
        awake = torch.zeros(0, dtype=torch.long, device=dev)         # state column -> global
        v = torch.zeros((B, 0), device=dev)
        g = torch.zeros((B, 0), device=dev)
        last_spike = torch.zeros((B, 0), dtype=torch.long, device=dev)
        rfc = torch.zeros(0, dtype=torch.long, device=dev)
        ring = torch.zeros((delay_steps + 1, B, N), device=dev)  # future synaptic input

        def wake(new):
            nonlocal awake, v, g, last_spike, rfc
            new = torch.unique(new)
            new = new[local[new] < 0]
            if len(new) == 0:
                return
            local[new] = torch.arange(len(awake), len(awake) + len(new), device=dev)
            awake = torch.cat([awake, new])
            v = torch.cat([v, torch.full((B, len(new)), p.v_rest, device=dev)], 1)
            g = torch.cat([g, torch.zeros((B, len(new)), device=dev)], 1)
            last_spike = torch.cat(
                [last_spike, torch.full((B, len(new)), -10**9, dtype=torch.long, device=dev)], 1)
            rfc = torch.cat([rfc, torch.full((len(new),), ref_steps, dtype=torch.long, device=dev)])

        # Stimulated neurons have no refractory period (as in the original model).
        stim_global, stim_prob = [], []
        for s in stimuli:
            if s.rate_hz <= 0 or not s.indices:
                continue
            idx = torch.tensor(s.indices, device=dev)
            wake(idx)
            stim_global.append(idx)
            stim_prob.append(s.rate_hz * p.dt / 1000.0)
        for idx in stim_global:
            rfc[local[idx]] = 0

        rec_steps, rec_trials, rec_neurons = [], [], []
        t0 = time.time()
        for t in range(n_steps):
            # 1) integrate everyone who is not refractory
            free = (t - last_spike) >= rfc
            v_new = p.v_rest + (v - p.v_rest) * a_v + g * c_g
            v = torch.where(free, v_new, v)
            g = torch.where(free, g * a_g, g)

            # 2) threshold -> spikes
            spiked = (v > p.v_threshold) & free

            # 3) synaptic input that was sent `delay` ago arrives now.
            #    A refractory neuron ignores it (same as Brian2 for "unless refractory").
            slot = t % (delay_steps + 1)
            g += ring[slot][:, awake] * free
            ring[slot][:, awake] = 0.0

            # 4) stimulus: Poisson kicks to the driven neurons
            for idx, prob in zip(stim_global, stim_prob):
                hit = torch.rand((B, len(idx)), generator=gen, device=dev) < prob
                v[:, local[idx]] += hit * p.poisson_weight_mV

            # 5) send this step's spikes down every outgoing synapse
            b_idx, c_idx = spiked.nonzero(as_tuple=True)
            if len(c_idx):
                n_idx = awake[c_idx]
                # reset the neurons that spiked (before `wake` may grow the state)
                v[b_idx, c_idx] = p.v_reset
                g[b_idx, c_idx] = 0.0
                last_spike[b_idx, c_idx] = t
                starts = self.indptr[n_idx]
                counts = self.indptr[n_idx + 1] - starts
                total = int(counts.sum())
                if total:
                    offsets = torch.cumsum(counts, 0) - counts
                    syn = (torch.arange(total, device=dev)
                           - torch.repeat_interleave(offsets, counts)
                           + torch.repeat_interleave(starts, counts))
                    b_rep = torch.repeat_interleave(b_idx, counts)
                    targets = self.post[syn]
                    wake(targets)
                    target = (t + delay_steps) % (delay_steps + 1)
                    ring[target].index_put_((b_rep, targets), weight[syn], accumulate=True)
                rec_steps.append(torch.full_like(n_idx, t))
                rec_trials.append(b_idx)
                rec_neurons.append(n_idx)

            if verbose and t and t % (n_steps // 10) == 0:
                print(f"  {100 * t // n_steps:3d}%  ({time.time() - t0:.1f}s)")

        cat = (lambda xs: torch.cat(xs).cpu().numpy()) if rec_steps else (
            lambda xs: np.zeros(0, dtype=np.int64))
        return SimResult(
            steps=cat(rec_steps), trials=cat(rec_trials), neurons=cat(rec_neurons),
            params=p, n_neurons=N, wall_time_s=time.time() - t0,
            meta={"stimuli": [(s.name, s.rate_hz) for s in stimuli],
                  "silenced": list(silence or [])},
        )
