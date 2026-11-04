import json
import torch
import numpy as np
from data.pino_data_pipeline import PINONormalizer, load_full_split_tensor
from data.store import open_project_zarr
from train.train_baselines import calculate_metrics

def main():
    with open("DATASET/norm_stats.json") as f:
        stats = json.load(f)
    normalizer = PINONormalizer(stats)

    train_ds = open_project_zarr("DATASET/train_2018_2019.zarr")
    test_ds = open_project_zarr("DATASET/test_2021_2022.zarr")

    train_raw = load_full_split_tensor(train_ds)
    test_raw = load_full_split_tensor(test_ds)

    lats = train_ds.latitude.values
    train_clim_overall = train_raw.mean(dim=0)

    train_hours = np.asarray(train_ds.time.dt.hour.values)
    test_hours = np.asarray(test_ds.time.dt.hour.values)

    diurnal_clim = {}
    for h in [0, 6, 12, 18]:
        mask = train_hours == h
        diurnal_clim[h] = train_raw[mask].mean(dim=0)

    targets = test_raw[1:]
    inputs = test_raw[:-1]
    tgt_hours = test_hours[1:]
    inp_hours = test_hours[:-1]

    res_pers = calculate_metrics(inputs, targets, normalizer, lats, train_clim_overall)

    diurnal_preds = torch.stack([diurnal_clim[int(h)] for h in tgt_hours])
    res_diurnal_clim = calculate_metrics(diurnal_preds, targets, normalizer, lats, train_clim_overall)

    inp_clim = torch.stack([diurnal_clim[int(h)] for h in inp_hours])
    anom = inputs - inp_clim
    diurnal_anom_pers_preds = diurnal_preds + anom
    res_diurnal_anom_pers = calculate_metrics(diurnal_anom_pers_preds, targets, normalizer, lats, train_clim_overall)

    print("=== 1-Step (6h) Non-Learned Baselines ===")
    print("Standard Persistence:          nRMSE = {:.4f} | ACC = {:.4f} | T2m RMSE = {:.4f} K".format(
        res_pers["aggregate_nrmse"], res_pers["aggregate_acc"], res_pers["rmse"]["T2m"]))
    print("Diurnal Climatology:           nRMSE = {:.4f} | ACC = {:.4f} | T2m RMSE = {:.4f} K".format(
        res_diurnal_clim["aggregate_nrmse"], res_diurnal_clim["aggregate_acc"], res_diurnal_clim["rmse"]["T2m"]))
    print("Diurnal + Anomaly Persistence: nRMSE = {:.4f} | ACC = {:.4f} | T2m RMSE = {:.4f} K".format(
        res_diurnal_anom_pers["aggregate_nrmse"], res_diurnal_anom_pers["aggregate_acc"], res_diurnal_anom_pers["rmse"]["T2m"]))

    print("\n=== Multi-Horizon Non-Learned Baselines (nRMSE) ===")
    print("{:<7s} | {:<13s} | {:<12s} | {:<20s}".format("Horizon", "Standard Pers", "Diurnal Clim", "Diurnal+Anom Pers"))
    print("-" * 60)
    horizons_data = {}
    for step, h_name in [(1, "6h"), (2, "12h"), (4, "24h"), (8, "48h"), (12, "72h")]:
        tgt = test_raw[step:]
        inp = test_raw[:-step]
        t_hrs = test_hours[step:]
        i_hrs = test_hours[:-step]

        m_p = calculate_metrics(inp, tgt, normalizer, lats, train_clim_overall)
        d_p = torch.stack([diurnal_clim[int(h)] for h in t_hrs])
        m_dc = calculate_metrics(d_p, tgt, normalizer, lats, train_clim_overall)
        i_c = torch.stack([diurnal_clim[int(h)] for h in i_hrs])
        da_p = d_p + (inp - i_c)
        m_dap = calculate_metrics(da_p, tgt, normalizer, lats, train_clim_overall)

        horizons_data[h_name] = {
            "persistence": m_p["aggregate_nrmse"],
            "diurnal_clim": m_dc["aggregate_nrmse"],
            "diurnal_anom_pers": m_dap["aggregate_nrmse"],
            "persistence_t2m": m_p["rmse"]["T2m"],
            "diurnal_anom_pers_t2m": m_dap["rmse"]["T2m"],
        }
        print("{:<7s} | {:<13.4f} | {:<12.4f} | {:<20.4f}".format(
            h_name, m_p["aggregate_nrmse"], m_dc["aggregate_nrmse"], m_dap["aggregate_nrmse"]))

    train_ds.close()
    test_ds.close()

    with open("DATASET/stronger_baselines.json", "w") as f:
        json.dump(horizons_data, f, indent=2)

if __name__ == "__main__":
    main()

