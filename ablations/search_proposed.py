"""Budget-aware, resumable search for proposed-model ablations.

All nine configurations receive a short validation screen. Only the best
``--top-k`` configurations receive full training and test/rollout evaluation.
Progress is saved after every configuration, so an interrupted run resumes
without spending credits on completed work.
"""

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import PINONormalizer, get_rollout_loader, get_single_step_loader
from train.proposed_model import ProposedModel
from train.train_baselines import (
    evaluate_model_on_test,
    evaluate_rollout_horizons,
    fine_tune_rollout,
    save_checkpoint,
    train_one_epoch,
    validate,
)


CONFIGS = [
    {"physics_mode": mode, "use_gnn_residual": use_gnn, "gnn_dense": dense,
     "refinement_steps": 3, "refinement_alpha": 0.5}
    for mode in ("none", "soft", "hard")
    for use_gnn, dense in ((False, False), (True, False), (True, True))
]
STATE_PATH = "DATASET/search_state.json"
RESULTS_PATH = "DATASET/proposed_results.json"


def config_name(config):
    variant = "Dense" if config["gnn_dense"] else "Sparse" if config["use_gnn_residual"] else "Off"
    return f"FactorizedFNO_GNN{variant}_{config['physics_mode']}_R{config['refinement_steps']}"


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def time_expired(start, budget_seconds, reserve_seconds=0):
    return time.perf_counter() - start >= max(0, budget_seconds - reserve_seconds)


def screen_config(config, terrain_path, train_loader, val_loader, normalizer, device,
                  start, budget_seconds, epochs, patience):
    model = ProposedModel(terrain_path=terrain_path, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    use_soft = config["physics_mode"] == "soft"
    best_val, no_improve = float("inf"), 0
    try:
        for epoch in range(1, epochs + 1):
            if time_expired(start, budget_seconds, reserve_seconds=30):
                break
            train_one_epoch(model, train_loader, optimizer, device,
                            use_physics_loss=use_soft, lambda_div=0.01,
                            lambda_hydro=0.1, normalizer=normalizer)
            val_loss, _ = validate(model, val_loader, device,
                                   use_physics_loss=use_soft, lambda_div=0.01,
                                   lambda_hydro=0.1, normalizer=normalizer)
            print(f"  screen epoch {epoch}/{epochs}: val={val_loss:.5f}", flush=True)
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return best_val


def train_full(config, terrain_path, train_loader, val_loader, test_loader,
               train_rollout, test_rollout, normalizer, climatology, device,
               start, budget_seconds, epochs, patience, rollout_epochs):
    name = config_name(config)
    model = ProposedModel(terrain_path=terrain_path, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    use_soft = config["physics_mode"] == "soft"
    best_state, best_val, no_improve = None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        if time_expired(start, budget_seconds, reserve_seconds=30):
            raise TimeoutError(f"Budget exhausted before completing {name}; resume later.")
        train_one_epoch(model, train_loader, optimizer, device,
                        use_physics_loss=use_soft, lambda_div=0.01,
                        lambda_hydro=0.1, normalizer=normalizer)
        val_loss, _ = validate(model, val_loader, device,
                               use_physics_loss=use_soft, lambda_div=0.01,
                               lambda_hydro=0.1, normalizer=normalizer)
        print(f"  full epoch {epoch}/{epochs}: val={val_loss:.5f}", flush=True)
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint was produced for {name}")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    print(f"  checkpoint: {save_checkpoint(best_state, name)}", flush=True)
    if rollout_epochs:
        fine_tune_rollout(model, train_rollout,
                          torch.optim.AdamW(model.parameters(), lr=1e-4),
                          device, rollout_epochs)
    return {
        "one_step": evaluate_model_on_test(model, test_loader, normalizer, device, climatology=climatology),
        "rollout": evaluate_rollout_horizons(model, test_rollout, normalizer, device, climatology=climatology),
        "config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_val_loss": best_val,
    }


def run(budget_hours=3.0, top_k=3, screen_epochs=8, full_epochs=30,
        screen_patience=3, full_patience=6, rollout_epochs=3):
    start = time.perf_counter()
    budget_seconds = budget_hours * 3600.0
    terrain_path = "DATASET/terrain_hp.npz"
    required = [terrain_path, "DATASET/norm_stats.json",
                "DATASET/train_2018_2019.zarr", "DATASET/val_2020.zarr",
                "DATASET/test_2021_2022.zarr"]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = PINONormalizer(load_json("DATASET/norm_stats.json", {}))
    train_loader = get_single_step_loader(required[2], normalizer, 32, True)
    val_loader = get_single_step_loader(required[3], normalizer, 32, False)
    test_loader = get_single_step_loader(required[4], normalizer, 32, False)
    train_rollout = get_rollout_loader(required[2], normalizer, 4, 8, True)
    test_rollout = get_rollout_loader(required[4], normalizer, 13, 8, False)
    climatology = normalizer.denormalize(train_loader.dataset.data.float()).mean(dim=0)

    state = load_json(STATE_PATH, {"screening": {}, "full": {}})
    results = load_json(RESULTS_PATH, {})
    state.setdefault("screening", {})
    state.setdefault("full", {})

    for config in CONFIGS:
        name = config_name(config)
        if name in state["screening"]:
            continue
        if time_expired(start, budget_seconds, reserve_seconds=30):
            save_json(STATE_PATH, state)
            print("[BUDGET] Stopping before another screening run; resume later.", flush=True)
            return
        print(f"[SCREEN] {name}", flush=True)
        state["screening"][name] = screen_config(
            config, terrain_path, train_loader, val_loader, normalizer, device,
            start, budget_seconds, screen_epochs, screen_patience)
        save_json(STATE_PATH, state)

    ranked = sorted(state["screening"].items(), key=lambda item: item[1])
    survivors = {name for name, _ in ranked[:max(1, min(top_k, len(ranked)))]}
    print(f"[RANK] Survivors: {sorted(survivors)}", flush=True)
    config_by_name = {config_name(config): config for config in CONFIGS}

    for name in [name for name, _ in ranked if name in survivors]:
        if name in state["full"] and name in results:
            continue
        if time_expired(start, budget_seconds, reserve_seconds=60):
            save_json(STATE_PATH, state)
            save_json(RESULTS_PATH, results)
            print("[BUDGET] Stopping before another full run; resume later.", flush=True)
            return
        print(f"[FULL] {name}", flush=True)
        result = train_full(config_by_name[name], terrain_path, train_loader, val_loader,
                            test_loader, train_rollout, test_rollout, normalizer,
                            climatology, device, start, budget_seconds, full_epochs,
                            full_patience, rollout_epochs)
        results[name] = result
        state["full"][name] = True
        save_json(STATE_PATH, state)
        save_json(RESULTS_PATH, results)
    print("[DONE] Budget-aware proposed-model search complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-hours", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--full-epochs", type=int, default=30)
    parser.add_argument("--rollout-epochs", type=int, default=3)
    args = parser.parse_args()
    run(**vars(args))
