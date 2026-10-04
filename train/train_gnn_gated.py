"""Train GNN-sparse with soft physics + learnable gate. Falsifiable test."""
import json, sys, torch
sys.path.insert(0, '.')
from data.pino_data_pipeline import PINONormalizer, get_single_step_loader
from train.proposed_model import ProposedModel
from train.train_baselines import train_one_epoch, validate, save_checkpoint

DEVICE = torch.device('cuda')
TERRAIN = 'DATASET/terrain_hp.npz'
normalizer = PINONormalizer(json.load(open('DATASET/norm_stats.json')))
train_loader = get_single_step_loader('DATASET/train_2018_2019.zarr', normalizer, batch_size=16, shuffle=True)
val_loader   = get_single_step_loader('DATASET/val_2020.zarr', normalizer, batch_size=16, shuffle=False)

model = ProposedModel(terrain_path=TERRAIN, physics_mode='soft',
                      use_gnn_residual=True, gnn_dense=False,
                      refinement_steps=3, refinement_alpha=0.5).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5)

best_val, best_state, no_improve = float('inf'), None, 0
for epoch in range(1, 16):
    train_one_epoch(model, train_loader, opt, DEVICE,
                    use_physics_loss=True, lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer)
    val_loss, _ = validate(model, val_loader, DEVICE,
                           use_physics_loss=True, lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer)
    sched.step()
    gate_val = model.gnn.gnn_scale.item()
    mark = ' *' if val_loss < best_val else ''
    print(f'  epoch {epoch:2d}/15  val={val_loss:.5f}  gate={gate_val:.6f}{mark}', flush=True)
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= 6:
        print(f'  [Early Stop] best_val={best_val:.5f}', flush=True)
        break

save_checkpoint(best_state, 'GNNSparse_soft_gated')
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
final_gate = model.gnn.gnn_scale.item()
print(f'\nFinal gnn_scale gate: {final_gate:.6f}', flush=True)
if final_gate > 0.05:
    print('[GATE VERDICT] Gate OPENED — GNN has genuine terrain signal.', flush=True)
else:
    print('[GATE VERDICT] Gate stayed CLOSED — confirms scale-mismatch. Reframe claim.', flush=True)
