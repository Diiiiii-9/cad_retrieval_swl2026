"""Shared fixtures: a tiny synthetic dataset of raw CAD graphs (same format as data_converter.py)."""
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_raw_graph(seed, num_faces=None, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    n = num_faces or int(torch.randint(6, 16, (1,), generator=g))
    x = torch.stack([
        torch.rand(n, generator=g) * 100 * scale,                 # area
        torch.randint(0, 10, (n,), generator=g).float(),          # surface type
        torch.randint(0, 2, (n,), generator=g).float(),           # closed u
        torch.randint(0, 2, (n,), generator=g).float(),           # closed v
    ], dim=1)
    src, dst, feats = [], [], []
    for i in range(n):                      # ring + a few chords, stored in both directions
        for j in {(i + 1) % n, (i + 3) % n}:
            f = [float(torch.rand(1, generator=g)) * 10 * scale, float(torch.randint(0, 8, (1,), generator=g)), 0.0]
            src += [i, j]; dst += [j, i]; feats += [f, f]
    return Data(x=x, edge_index=torch.tensor([src, dst]), edge_attr=torch.tensor(feats))


@pytest.fixture
def graph_dir(tmp_path):
    d = tmp_path / "graphs"
    d.mkdir()
    for i in range(48):
        cls = i % 3   # MFCAD-style names: <class>-<subclass>-...-<id>
        torch.save(make_raw_graph(i), d / f"{cls}-{i % 2}-0-{i}.pt")
    return d
