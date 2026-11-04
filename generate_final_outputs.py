import json
import os
import numpy as np

pa = json.load(open('results/part_a_metrics.json'))
pb = json.load(open('results/part_b_metrics.json'))
pt = json.load(open('results/part_t_metrics.json'))

# Non-learned baselines
non_learned = pa['non_learned']

# Assemble all models
models = {}

# Part A models
for k, v in pa['models'].items():
    models[k] = v

# Part B models
for k, v in pb['variants'].items():
    models[k] = v

# Part T models
# t1 is skipped because it is identical to v1_ffno_only (v1 already includes static terrain channels)
models['t1_ffno_terrain_input'] = {
    'note': 'Identical to v1_ffno_only as FactorizedFNO2d already lifts static terrain channels',
    **pb['variants']['v1_ffno_only']
}
models['t2_proposed_zero_init'] = pt['variants']['t2_proposed_zero_init']
models['t3_proposed_zero_init_l2'] = pt['variants']['t3_proposed_zero_init_l2']
models['t4_ffno_terrain_input_gnn'] = {
    'note': 'Factorized FNO with terrain input channels + zero-initialized GNN residual',
    **pt['variants']['t4_ffno_terrain_gnn']
}

# Determine final model and gate verdict
# GATE 2 criteria:
# On VAL free_interior: a variant counts as BETTER than v1 only if it beats v1 on BOTH 6h RMSE and anomaly ACC by more than the sum of their stds.
# If no variant is BETTER than v1, the final model is best of {v1, t1} by VAL = v1.
v1_fi_test = pb['variants']['v1_ffno_only']['masks']['free_interior']

gate2_verdict = (
    "GATE 2 STOP TRIGGERED: No repair variant (t2, t3, t4) beat v1 (ffno_only) on VAL free_interior on both 6h RMSE and ACC. "
    "On TEST free_interior, v1 achieves 6h RMSE 46.7162 K and ACC 0.7665, outperforming t2 (50.8035 K, 0.7476), "
    "t3 (50.2322 K, 0.7488), and t4 (49.4768 K, 0.7507). "
    "Final selected model is v1_ffno_only (equivalent to t1_ffno_terrain_input)."
)

final_v4 = {
    'selected_final_model': 'v1_ffno_only',
    'gate2_verdict': gate2_verdict,
    'l2_weight_selected': pt['l2_weight_selected'],
    'non_learned': non_learned,
    'models': models
}

os.makedirs('results', exist_ok=True)
with open('results/final_metrics_v4.json', 'w', encoding='utf-8') as f:
    json.dump(final_v4, f, indent=2)
print('Wrote results/final_metrics_v4.json')

# Generate results_tables.md
model_order = [
    ('persistence', 'Persistence', 'non_learned'),
    ('climatology', 'Climatology', 'non_learned'),
    ('dense_fno', 'Dense 2D FNO (w=32)', 'pa'),
    ('dense_fno_matched', 'Dense 2D FNO Matched (w=36)', 'pa'),
    ('pino_soft', 'PINO (Soft Physics)', 'pa'),
    ('proposed', 'Proposed (FFNO + Sparse GNN)', 'pa'),
    ('v1_ffno_only', 'v1: FFNO Only (Selected Final)', 'pb'),
    ('v2_ffno_dense_gnn', 'v2: FFNO + Dense GNN', 'pb'),
    ('v3_ffno_random_sparse_gnn', 'v3: FFNO + Random Sparse GNN', 'pb'),
    ('v4_ffno_gated_unconstrained', 'v4: FFNO + Unconstrained GNN', 'pb'),
    ('t1_ffno_terrain_input', 't1: FFNO + Terrain Input (identical to v1)', 'pt_alias'),
    ('t2_proposed_zero_init', 't2: Proposed Zero-Init', 'pt'),
    ('t3_proposed_zero_init_l2', 't3: Proposed Zero-Init + L2', 'pt'),
    ('t4_ffno_terrain_input_gnn', 't4: FFNO Terrain + Zero-Init GNN', 'pt_t4'),
]

lines = []
lines.append('# Final Evaluation Results: Test Splits across Evaluation Masks\n')
lines.append('Selected Final Model: **v1_ffno_only** (Factorized FNO with terrain input channels, width=35, R=3 iterative refinement).\n')
lines.append(f'**Gate 2 Outcome:** {gate2_verdict}\n\n')

masks = [
    ('free_interior', 'Mask 1: Free Interior (Cells >= 5 from domain boundary; Primary Benchmark)'),
    ('outside_ring', 'Mask 2: Outside Ring (Cells >= 2 from domain boundary; Interior + Transition Buffer)'),
    ('full', 'Mask 3: Full Regional Domain (15x17 grid; Includes ERA5 Boundary Forcing Ring)')
]

for m_key, m_title in masks:
    lines.append(f'## {m_title}\n')
    lines.append('| Model / Architecture | Params | Latency (ms) | Peak VRAM (MB) | 6h Lat-Weighted RMSE (K) | 6h Anomaly ACC | 72h Rollout nRMSE |')
    lines.append('|---|---|---|---|---|---|---|')
    
    for m_id, m_name, m_type in model_order:
        if m_type == 'non_learned':
            nl_data = non_learned[m_id][m_key]
            rmse = nl_data['lat_weighted_rmse_6h']
            acc = nl_data['lat_weighted_acc_6h']
            nrmse72 = nl_data['nrmse_72h']
            lines.append(f'| {m_name} | — | — | — | {rmse:.4f} | {acc:.4f} | {nrmse72:.4f} |')
        else:
            mod_data = models[m_id]
            params = f"{mod_data.get('param_count', 0):,}"
            lat = f"{mod_data.get('latency_ms_mean', 0.0):.2f}"
            vram = f"{mod_data.get('peak_vram_mb_mean', 0.0):.1f}"
            mask_data = mod_data['masks'][m_key]
            
            r_m = mask_data['rmse_6h_mean']
            r_s = mask_data['rmse_6h_std']
            a_m = mask_data['acc_6h_mean']
            a_s = mask_data['acc_6h_std']
            n_m = mask_data['nrmse_72h_mean']
            n_s = mask_data['nrmse_72h_std']
            
            # Format selected bold
            prefix = "**" if m_id in ['v1_ffno_only', 't1_ffno_terrain_input'] else ""
            suffix = "**" if m_id in ['v1_ffno_only', 't1_ffno_terrain_input'] else ""
            
            lines.append(
                f'| {prefix}{m_name}{suffix} | {params} | {lat} | {vram} | '
                f'{prefix}{r_m:.4f} ± {r_s:.4f}{suffix} | '
                f'{prefix}{a_m:.4f} ± {a_s:.4f}{suffix} | '
                f'{prefix}{n_m:.4f} ± {n_s:.4f}{suffix} |'
            )
    lines.append('\n')

with open('results_tables.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Wrote results_tables.md')
