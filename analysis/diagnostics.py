import json
import os
import random

import numpy as np
import torch

from data.pino_data_pipeline import PINONormalizer, get_single_step_loader
from models.proposed_model import ProposedModel
from train.train_baselines import compute_divergence_loss, compute_hydrostatic_loss


SEED = 2026
DATASET_DIR = "DATASET"
CHECKPOINT_NAME = "FactorizedFNO_GNNOff_soft_R3"
CHANNELS = ("T2m", "U10", "V10", "Z500", "T850")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_saved_results():
    with open(os.path.join(DATASET_DIR, "baseline_results.json"), encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(os.path.join(DATASET_DIR, "proposed_results.json"), encoding="utf-8") as handle:
        proposed = json.load(handle)

    rows = [
        (name, result)
        for name, result in baseline.items()
        if "aggregate_nrmse" in result and "aggregate_acc" in result
    ]
    for name, result in proposed.items():
        rows.append((name, result["one_step"]))

    print("SAVED RESULTS")
    for name, result in rows:
        print(f"{name}: aggregate_nRMSE={result['aggregate_nrmse']:.6f}, aggregate_ACC={result['aggregate_acc']:.6f}")
        print("  " + ", ".join(f"{channel}_RMSE={result['rmse'][channel]:.6f}" for channel in CHANNELS))


def print_physics_diagnostics():
    with open(os.path.join(DATASET_DIR, "norm_stats.json"), encoding="utf-8") as handle:
        normalizer = PINONormalizer(json.load(handle))

    loader = get_single_step_loader(
        os.path.join(DATASET_DIR, "test_2021_2022.zarr"),
        normalizer,
        batch_size=8,
        shuffle=False,
    )
    inputs, targets = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposedModel(
        terrain_path=os.path.join(DATASET_DIR, "terrain_hp.npz"),
        physics_mode="soft",
        use_gnn_residual=False,
        refinement_steps=3,
        refinement_alpha=0.5,
    ).to(device)
    checkpoint = os.path.join(DATASET_DIR, "checkpoints", CHECKPOINT_NAME + ".pt")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    inputs = inputs.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        prediction = model(inputs)

    truth_raw = normalizer.denormalize(targets.float())
    prediction_raw = normalizer.denormalize(prediction.float())
    print("PHYSICS DIAGNOSTICS")
    print(f"device={device}, batch_shape={tuple(targets.shape)}")
    print(f"truth_divergence_loss={compute_divergence_loss(truth_raw, None).item():.12g}")
    print(f"prediction_divergence_loss={compute_divergence_loss(prediction_raw, None).item():.12g}")
    print(f"truth_hydrostatic_loss={compute_hydrostatic_loss(truth_raw, None).item():.12g}")
    print(f"prediction_hydrostatic_loss={compute_hydrostatic_loss(prediction_raw, None).item():.12g}")


if __name__ == "__main__":
    set_seed(SEED)
    print_saved_results()
    print_physics_diagnostics()
