"""
Graph Sanity Check
==================
Scans a folder of processed .pt graphs for problems that break training:
NaN/inf features, empty graphs, graphs without edges, unexpected feature sizes.

Usage:
    python scripts/check_graphs.py data/processed_graphs/fusion360
"""

import sys
from collections import Counter
from pathlib import Path

import torch


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    files = sorted(Path(sys.argv[1]).glob("*.pt"))
    issues, shapes = {}, Counter()
    for f in files:
        d = torch.load(f, weights_only=False)
        problems = []
        if d.x is None or d.x.shape[0] == 0:
            problems.append("no faces")
        else:
            shapes[(d.x.shape[1], d.edge_attr.shape[1] if d.edge_attr is not None and d.edge_attr.dim() == 2 else 0)] += 1
            if not torch.isfinite(d.x).all():
                problems.append(f"non-finite face features in columns {sorted(set(torch.nonzero(~torch.isfinite(d.x))[:, 1].tolist()))}")
        if d.edge_index is None or d.edge_index.numel() == 0:
            problems.append("no edges")
        elif d.edge_attr is not None and not torch.isfinite(d.edge_attr).all():
            problems.append(f"non-finite edge features in columns {sorted(set(torch.nonzero(~torch.isfinite(d.edge_attr))[:, 1].tolist()))}")
        if problems:
            issues[f.stem] = problems

    print(f"Scanned {len(files)} graphs. Feature sizes (node, edge): {dict(shapes)}")
    kinds = Counter(p.split(" in ")[0] for ps in issues.values() for p in ps)
    print(f"Graphs with issues: {len(issues)}  {dict(kinds)}")
    for name, ps in list(issues.items())[:20]:
        print(f"  {name}: {'; '.join(ps)}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")
    print("\nNon-finite values are replaced by 0 when loading (src/dataset.py), so training is safe either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
