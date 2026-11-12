"""End-to-end smoke test: train -> build FAISS index -> evaluate, on synthetic graphs (CPU, ~seconds)."""
import json
from pathlib import Path

import pytest

faiss = pytest.importorskip("faiss")

from src import train_unsupervised, retrieval
from src.fusion360 import train_unsupervised_fusion360, retrieval_fusion360
from src.run_config import load_run_config
import scripts.evaluate_retrieval as ev

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("features,pooling", [("raw", "mean"), ("normalized", "max_mean")])
def test_train_build_evaluate(graph_dir, tmp_path, features, pooling):
    run = tmp_path / f"run_{features}_s0"
    train_unsupervised.train(["--data_dir", str(graph_dir), "--save_dir", str(run), "--epochs", "2",
                              "--batch_size", "8", "--features", features, "--pooling", pooling,
                              "--out_dim", "16", "--hidden_dim", "32"])
    cfg, found = load_run_config(run)
    assert found and cfg["features"] == features and cfg["pooling"] == pooling and cfg["out_dim"] == 16
    assert len(json.loads((run / "train_history.json").read_text())) == 2

    # Index building must pick up out_dim/pooling/features from config.json (CLI defaults differ!)
    retrieval.main(["--build", "--data_dir", str(graph_dir), "--save_dir", str(run)])
    index = faiss.read_index(str(run / "faiss_index.bin"))
    assert index.ntotal == 48 and index.d == 16

    out = tmp_path / "res"
    assert ev.main(["--dataset", "mfcad", "--runs", str(run), "--graphs_dir", str(graph_dir), "--out", str(out)]) == 0
    agg = json.loads(out.with_suffix(".json").read_text())["aggregated"]
    configs = {a["config"] for a in agg}
    assert {f"run_{features}", "baseline_random", "baseline_handcrafted"} <= configs


def test_fusion360_wrappers(graph_dir, tmp_path):
    run = tmp_path / "f360"
    train_unsupervised_fusion360.train(["--data_dir", str(graph_dir), "--save_dir", str(run), "--epochs", "1",
                                        "--batch_size", "8", "--features", "normalized"])
    retrieval_fusion360.main(["--build", "--data_dir", str(graph_dir), "--save_dir", str(run)])
    assert (run / "faiss_index_fusion360.bin").exists() and (run / "metadata_fusion360.pkl").exists()


def test_legacy_checkpoint_still_loads():
    """Checkpoints trained before config.json existed must keep working with the defaults."""
    import torch
    from src.networks import CADGraphEncoder
    from src.run_config import load_encoder_weights, LEGACY_DEFAULTS
    ckpt = ROOT / "checkpoints"
    if not (ckpt / "final_encoder_weights.pth").exists():
        pytest.skip("no committed checkpoint")
    enc = CADGraphEncoder(4, 3, hidden_dim=LEGACY_DEFAULTS["hidden_dim"], out_dim=LEGACY_DEFAULTS["out_dim"])
    load_encoder_weights(enc, ckpt, torch.device("cpu"))


def test_nan_graph_does_not_break_training(graph_dir, tmp_path):
    """One graph with NaN/inf geometry (as found in Fusion 360) must not turn the model into NaN."""
    import math
    import torch
    from tests.conftest import make_raw_graph
    bad = make_raw_graph(999)
    bad.x[0, 0] = float("nan")
    bad.x[1, 0] = float("inf")
    bad.edge_attr[0, 0] = float("inf")
    torch.save(bad, graph_dir / "0-0-0-999.pt")

    for features in ("raw", "normalized"):
        run = tmp_path / f"nan_{features}"
        train_unsupervised.train(["--data_dir", str(graph_dir), "--save_dir", str(run), "--epochs", "2",
                                  "--batch_size", "8", "--features", features])
        history = json.loads((run / "train_history.json").read_text())
        assert all(math.isfinite(h["loss"]) for h in history)
        retrieval.main(["--build", "--data_dir", str(graph_dir), "--save_dir", str(run)])
        index = faiss.read_index(str(run / "faiss_index.bin"))
        import numpy as np
        assert np.isfinite(index.reconstruct_n(0, index.ntotal)).all()


# Added 2026-11-11 by Eduardo Dall Igna: input-ablation (--mask) tests
def test_mask_is_used_for_index_building(graph_dir, tmp_path):
    """The index must be built with the SAME mask the model was trained with (read from config.json),
    otherwise the encoder sees information at retrieval time that it never saw in training."""
    run = tmp_path / "masked_s0"
    train_unsupervised.train(["--data_dir", str(graph_dir), "--save_dir", str(run), "--epochs", "1",
                              "--batch_size", "8", "--features", "normalized", "--mask", "no_type"])
    cfg, found = load_run_config(run)
    assert found and cfg["mask"] == "no_type"

    args = retrieval.build_arg_parser().parse_args(["--build", "--save_dir", str(run)])
    from src.run_config import resolve_arch
    args, _ = resolve_arch(args, run)
    assert args.mask == "no_type"

    retrieval.main(["--build", "--data_dir", str(graph_dir), "--save_dir", str(run)])
    assert (run / "faiss_index.bin").exists()


def test_old_runs_without_mask_default_to_none(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"features": "raw", "out_dim": 64}))
    cfg, found = load_run_config(tmp_path)
    assert found and cfg["mask"] == "none"
