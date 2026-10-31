"""
Feature Normalization for CAD Graphs
====================================
@Description:
The raw graphs produced by data_converter.py store
    node x     = [area, surf_idx, closed_u, closed_v]          (4 dims)
    edge_attr  = [length, curve_idx, closed]                    (3 dims)

Two problems with feeding these directly into the GNN:
  1. `area` / `length` are in absolute model units and heavily skewed, so they
     dominate the input scale and make the model sensitive to the part's size.
  2. `surf_idx` / `curve_idx` are *categories* stored as numbers 0..9, which
     tells the network that "cone (2)" is between "cylinder (1)" and "sphere (3)".

NormalizeCADFeatures fixes both at load time (no STEP re-conversion needed):
    node x     = [area_frac, log_rel_area, one_hot(surf, 10), closed_u, closed_v]  (14 dims)
    edge_attr  = [len_frac, log_rel_len, one_hot(curve, 8), closed]                (11 dims)

  * area_frac    = area / total area of the part          (scale invariant)
  * log_rel_area = log(area / mean face area of the part) (scale invariant)
  * same idea for edge lengths.

The continuous columns are always first (NODE_CONT_DIMS / EDGE_CONT_DIMS), which
lets the contrastive augmentation add Gaussian noise ONLY to continuous features
instead of corrupting the one-hot category columns.
"""

import torch
import torch.nn.functional as F

NUM_SURF_TYPES = 10   # 9 named surface types + 'Other' (see data_converter.SURFACE_TYPES)
NUM_CURVE_TYPES = 8   # 7 named curve types + 'Other'   (see data_converter.CURVE_TYPES)

RAW_NODE_DIM, RAW_EDGE_DIM = 4, 3
NODE_CONT_DIMS, EDGE_CONT_DIMS = 2, 2
NORM_NODE_DIM = NODE_CONT_DIMS + NUM_SURF_TYPES + 2     # 14
NORM_EDGE_DIM = EDGE_CONT_DIMS + NUM_CURVE_TYPES + 1    # 11

FEATURE_MODES = ("raw", "normalized")
_EPS = 1e-8


def _relative_scale(values):
    """Returns (fraction of total, log ratio to mean) for a 1-D tensor of non-negative sizes."""
    values = values.clamp(min=0.0)
    if values.numel() == 0:
        return values.clone(), values.clone()
    total = values.sum()
    frac = values / (total + _EPS)
    log_rel = torch.log((values + _EPS) / (values.mean() + _EPS))
    # Degenerate faces/edges (size 0) would give log(eps/mean) -> very negative; clip it.
    log_rel = log_rel.clamp(min=-10.0, max=10.0)
    return frac, log_rel


def _one_hot(idx_col, num_classes):
    idx = idx_col.round().long().clamp(min=0, max=num_classes - 1)
    return F.one_hot(idx, num_classes=num_classes).float()


class NormalizeCADFeatures:
    """PyG-style transform: callable on a single (un-batched) Data object."""

    def __call__(self, data):
        data = data.clone()

        x = data.x.float()
        if x.shape[1] != RAW_NODE_DIM:
            raise ValueError(f"Expected raw node features with {RAW_NODE_DIM} dims, got {x.shape[1]}. "
                             "Is this graph already normalized?")
        area_frac, log_rel_area = _relative_scale(x[:, 0])
        data.x = torch.cat([
            area_frac.unsqueeze(1),
            log_rel_area.unsqueeze(1),
            _one_hot(x[:, 1], NUM_SURF_TYPES),
            x[:, 2:4],
        ], dim=1)

        ea = getattr(data, "edge_attr", None)
        if ea is None or ea.numel() == 0:
            data.edge_attr = torch.zeros((0, NORM_EDGE_DIM), dtype=torch.float)
        else:
            ea = ea.float()
            len_frac, log_rel_len = _relative_scale(ea[:, 0])
            data.edge_attr = torch.cat([
                len_frac.unsqueeze(1),
                log_rel_len.unsqueeze(1),
                _one_hot(ea[:, 1], NUM_CURVE_TYPES),
                ea[:, 2:3],
            ], dim=1)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_transform(features):
    """Maps the --features CLI value to a dataset transform (or None for raw)."""
    if features == "raw":
        return None
    if features == "normalized":
        return NormalizeCADFeatures()
    raise ValueError(f"Unknown feature mode {features!r}. Must be one of {FEATURE_MODES}.")


def noise_dims_for(features):
    """How many leading node-feature columns are continuous (and may receive Gaussian noise).
    None means 'all columns' (legacy behaviour for raw features)."""
    return NODE_CONT_DIMS if features == "normalized" else None
