"""Train and evaluate the factorized terrain-conditioned model ablations."""

import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import PINONormalizer, get_rollout_loader, get_single_step_loader, CHANNEL_NAMES
from train.proposed_model import FactorizedFNO2d, ProposedModel
from train.train_baselines import CoordinateFNO2d, evaluate_model_on_test, evaluate_rollout_horizons, fine_tune_rollout, save_checkpoint, train_one_epoch, validate


def run_proposed_experiments():
    terrain_path = os.path.join("DATASET", "terrain_hp.npz")
    if not os.path.exists(terrain_path):
        raise FileNotFoundError(
            f"Missing {terrain_path}. Generate it with data/terrain_pipeline.py before training."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join("DATASET", "norm_stats.json"), "r") as handle:
        normalizer = PINONormalizer(json.load(handle))

    train_loader = get_single_step_loader("DATASET/train_2018_2019.zarr", normalizer, 32, True)
    val_loader = get_single_step_loader("DATASET/val_2020.zarr", normalizer, 32, False)
    test_loader = get_single_step_loader("DATASET/test_2021_2022.zarr", normalizer, 32, False)
    train_rollout = get_rollout_loader("DATASET/train_2018_2019.zarr", normalizer, 4, 8, True)
    test_rollout = get_rollout_loader("DATASET/test_2021_2022.zarr", normalizer, 13, 8, False)
    climatology = normalizer.denormalize(train_loader.dataset.data.float()).mean(dim=0)
    results = {}
    baseline_params = sum(parameter.numel() for parameter in CoordinateFNO2d(in_dim=len(CHANNEL_NAMES), out_dim=len(CHANNEL_NAMES)).parameters())
    factorized_params = sum(parameter.numel() for parameter in FactorizedFNO2d().parameters())
    print(f"--- Step 0: Parameter count ---\n  CoordinateFNO2d: {baseline_params:,}\n  FactorizedFNO2d: {factorized_params:,}", flush=True)

    configs = [
        {"physics_mode": mode, "use_gnn_residual": use_gnn, "gnn_dense": dense}
        for mode in ("none", "soft", "hard")
        for use_gnn, dense in ((False, False), (True, False), (True, True))
    ]
    for config in configs:
        name = f"FactorizedFNO_GNN{'Dense' if config['gnn_dense'] else 'Sparse' if config['use_gnn_residual'] else 'Off'}_{config['physics_mode']}"
        print(f"\n--- Step 1: Training {name} ---", flush=True)
        model = ProposedModel(terrain_path=terrain_path, **config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best_state, best_val, no_improve = None, float("inf"), 0
        for epoch in range(1, 31):
            soft = config["physics_mode"] == "soft"
            train_values = train_one_epoch(
                model, train_loader, optimizer, device, use_physics_loss=soft,
                lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer,
            )
            val_loss, _ = validate(
                model, val_loader, device, use_physics_loss=soft,
                lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer,
            )
            print(f"  Epoch {epoch:02d}/30 -> Train: {train_values[0]:.5f} | Val: {val_loss:.5f}", flush=True)
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= 6:
                break
        assert best_state is not None, f"No checkpoint was saved for {name}"
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        print(f"  [Restored] Best weights for {name}", flush=True)
        print(f"  [Checkpoint] {save_checkpoint(best_state, name)}", flush=True)
        fine_tune_rollout(model, train_rollout, torch.optim.AdamW(model.parameters(), lr=1e-4), device, 3)
        results[name] = {
            "one_step": evaluate_model_on_test(model, test_loader, normalizer, device, climatology=climatology),
            "rollout": evaluate_rollout_horizons(model, test_rollout, normalizer, device, climatology=climatology),
            "config": config,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }

    with open(os.path.join("DATASET", "proposed_results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print("[SUCCESS] Proposed-model ablations saved to DATASET/proposed_results.json", flush=True)


if __name__ == "__main__":
    run_proposed_experiments()
