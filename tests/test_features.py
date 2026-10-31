import torch

from src.features import (NormalizeCADFeatures, NORM_NODE_DIM, NORM_EDGE_DIM, NODE_CONT_DIMS,
                          get_transform)
from src.train_unsupervised import augment_graph_batch
from tests.conftest import make_raw_graph


def test_normalized_shapes():
    data = NormalizeCADFeatures()(make_raw_graph(0))
    assert data.x.shape[1] == NORM_NODE_DIM
    assert data.edge_attr.shape[1] == NORM_EDGE_DIM
    assert torch.isfinite(data.x).all() and torch.isfinite(data.edge_attr).all()


def test_one_hot_matches_surface_type():
    raw = make_raw_graph(1)
    data = NormalizeCADFeatures()(raw)
    one_hot = data.x[:, NODE_CONT_DIMS:NODE_CONT_DIMS + 10]
    assert torch.equal(one_hot.argmax(1), raw.x[:, 1].long())
    assert torch.all(one_hot.sum(1) == 1)


def test_scale_invariance():
    """The same part modelled in mm or in m must get the same features."""
    t = NormalizeCADFeatures()
    a = t(make_raw_graph(2, num_faces=10, scale=1.0))
    b = t(make_raw_graph(2, num_faces=10, scale=1000.0))
    assert torch.allclose(a.x, b.x, atol=1e-4)
    assert torch.allclose(a.edge_attr, b.edge_attr, atol=1e-4)


def test_raw_mode_is_untouched():
    assert get_transform("raw") is None


def test_noise_only_on_continuous_columns():
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([NormalizeCADFeatures()(make_raw_graph(i)) for i in range(4)])
    aug = augment_graph_batch(batch, feature_noise_std=0.5, aug_mode="feature_noise", noise_dims=NODE_CONT_DIMS)
    assert not torch.equal(aug.x[:, :NODE_CONT_DIMS], batch.x[:, :NODE_CONT_DIMS])
    assert torch.equal(aug.x[:, NODE_CONT_DIMS:], batch.x[:, NODE_CONT_DIMS:])
    assert batch.x is not aug.x  # original batch must not be modified in place
