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
