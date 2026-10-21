"""
Connectome data: download and load the FlyWire whole-brain wiring diagram.

What is in the data?
--------------------
Two files published with Shiu et al. (2024, Nature):

* ``completeness.csv``   - one row per neuron (FlyWire ID). Row order defines the
                           integer index we use for every neuron in the simulation.
* ``connectivity.parquet`` - one row per (presynaptic, postsynaptic) neuron pair:
      Connectivity              number of synapses between the two neurons
      Excitatory                +1 (acetylcholine) or -1 (GABA / glutamate)
      Excitatory x Connectivity signed synapse count  -> the connection weight

We turn the connectivity table into a CSR ("compressed sparse row") matrix keyed
by the *presynaptic* neuron. When neuron ``i`` spikes, its outgoing synapses are
the slice ``indptr[i]:indptr[i+1]`` - this is what makes the simulation fast,
because each time step only touches the synapses of neurons that actually spiked.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_URL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"

# FlyWire release 630 is the version the paper used, so its neuron IDs match the
# published experiments (sugar / bitter / water neurons, MN9 ...).
DATASETS = {
    "630": {
        "completeness": "2023_03_23_completeness_630_final.csv",
        "connectivity": "2023_03_23_connectivity_630_final.parquet",
    },
    "783": {
        "completeness": "Completeness_783.csv",
        "connectivity": "Connectivity_783.parquet",
    },
}


def _fetch(url: str, path: Path, attempts: int = 3) -> None:
    """Download to a temporary file, check the size, then move it into place.

    A half-finished download never ends up under the real name, so an interrupted
    first run cannot leave a corrupt file behind.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                expected = int(r.headers.get("Content-Length") or 0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if expected and tmp.stat().st_size != expected:
                raise IOError(f"got {tmp.stat().st_size:,} of {expected:,} bytes")
            tmp.replace(path)
            return
        except Exception as e:  # network hiccup -> try again
            print(f"  download attempt {attempt} failed: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}")


def download(data_dir: str | Path = "data", version: str = "630") -> dict[str, Path]:
    """Download the connectome files (~90 MB) if they are not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, name in DATASETS[version].items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            _fetch(BASE_URL + name, path)
        paths[kind] = path
    return paths


@dataclass
class Connectome:
    """The whole brain as arrays the simulator can use directly."""

    flywire_ids: np.ndarray       # (N,) FlyWire ID of each neuron index
    indptr: torch.Tensor          # (N+1,) CSR row pointers (by presynaptic neuron)
    post: torch.Tensor            # (S,) postsynaptic neuron index of each connection
    weight: torch.Tensor          # (S,) signed synapse count of each connection
    sign: np.ndarray              # (N,) +1 excitatory, -1 inhibitory, 0 no outputs

    @property
    def n_neurons(self) -> int:
        return len(self.flywire_ids)

    @property
    def n_connections(self) -> int:
        return len(self.post)

    def index_of(self, flywire_ids) -> list[int]:
        """Convert FlyWire IDs to simulation indices."""
        lookup = self._lookup()
        missing = [f for f in flywire_ids if f not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} FlyWire IDs not in this connectome, e.g. {missing[0]}")
        return [lookup[f] for f in flywire_ids]

    def _lookup(self) -> dict[int, int]:
        if not hasattr(self, "_id2idx"):
            self._id2idx = {int(f): i for i, f in enumerate(self.flywire_ids)}
        return self._id2idx

    def outgoing(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Postsynaptic partners and weights of one neuron (handy for analysis)."""
        a, b = int(self.indptr[idx]), int(self.indptr[idx + 1])
        return self.post[a:b].numpy(), self.weight[a:b].numpy()


def load_connectome(data_dir: str | Path = "data", version: str = "630") -> Connectome:
    paths = download(data_dir, version)
    comp = pd.read_csv(paths["completeness"], index_col=0)
    con = pd.read_parquet(
        paths["connectivity"],
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    n = len(comp)
    pre = con["Presynaptic_Index"].to_numpy()
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = con["Postsynaptic_Index"].to_numpy()[order]
    w = con["Excitatory x Connectivity"].to_numpy()[order]

    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)

    # A neuron's transmitter decides the sign of ALL its outputs (Dale's principle).
    sign = np.zeros(n, dtype=np.int8)
    sign[pre] = np.sign(w)

    return Connectome(
        flywire_ids=comp.index.to_numpy(dtype=np.int64),
        indptr=torch.from_numpy(indptr),
        post=torch.from_numpy(post.astype(np.int64)),
        weight=torch.from_numpy(w.astype(np.float32)),
        sign=sign,
    )
