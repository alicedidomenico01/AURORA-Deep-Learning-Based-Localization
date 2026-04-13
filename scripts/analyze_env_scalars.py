#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utility metriche
# -------------------------
def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


# -------------------------
# Plot helpers (NO seaborn)
# -------------------------
def _savefig(outpath: Path, dpi=180):
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def plot_epoch_curves(summary_df: pd.DataFrame, outdir: Path, target_name: str):
    # summary_df: split, epoch, mae_*, rmse_*
    for metric in ["mae", "rmse", "r2", "corr"]:
        col = f"{metric}_{target_name}"
        if col not in summary_df.columns:
            continue
        plt.figure()
        for split, g in summary_df.groupby("split"):
            g = g.sort_values("epoch")
            plt.plot(g["epoch"].values, g[col].values, marker="o", label=split)
        plt.xlabel("epoch")
        plt.ylabel(col)
        plt.title(f"{col} over epochs")
        plt.legend()
        _savefig(outdir / f"{col}_over_epochs.png")


def plot_calibration_by_gt(df: pd.DataFrame, outdir: Path, pred_col: str, gt_col: str, name: str):
    # Raggruppa per valori discreti GT e plottami media±std del pred
    d = df[[pred_col, gt_col]].dropna()
    if d.empty:
        return

    grp = d.groupby(gt_col)[pred_col]
    gt_vals = np.array(sorted(grp.groups.keys()), dtype=float)
    means = np.array([grp.get_group(v).mean() for v in gt_vals], dtype=float)
    stds  = np.array([grp.get_group(v).std(ddof=0) for v in gt_vals], dtype=float)
    ns    = np.array([grp.get_group(v).shape[0] for v in gt_vals], dtype=int)

    plt.figure()
    plt.errorbar(gt_vals, means, yerr=stds, fmt="o-", capsize=4)
    # ideale: pred = gt
    minv = float(np.min(np.r_[gt_vals, means]))
    maxv = float(np.max(np.r_[gt_vals, means]))
    pad = 0.05 * (maxv - minv + 1e-6)
    lo, hi = minv - pad, maxv + pad
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(f"{gt_col} (discrete targets)")
    plt.ylabel(f"{pred_col} (model output)")
    plt.title(f"Calibration: {name}\n(points: mean±std, n shown in console)")
    _savefig(outdir / f"calibration_{name}.png")

    # Stampa tabellina utile
    calib = pd.DataFrame({"gt": gt_vals, "pred_mean": means, "pred_std": stds, "n": ns})
    calib.to_csv(outdir / f"calibration_{name}.csv", index=False)


def plot_hist_by_gt(
    df: pd.DataFrame,
    outdir: Path,
    pred_col: str,
    gt_col: str,
    name: str,
    bins: int = 40,
):
    d = df[[pred_col, gt_col]].dropna()
    if d.empty:
        return

    gt_vals = sorted(d[gt_col].unique().tolist())

    for v in gt_vals:
        plt.figure()

        x = d.loc[d[gt_col] == v, pred_col].values.astype(float)

        # ---- colore per tipo di scalare ----
        if name == "weather":
            hist_color = "seagreen"
        elif name == "illum":
            hist_color = "mediumpurple"
        else:
            hist_color = None

        plt.hist(
            x,
            bins=bins,
            color=hist_color,
            edgecolor="black",
            alpha=0.85,
        )

        plt.xlabel(pred_col)
        plt.ylabel("count")

        # ---- titolo semantico (solo estetica) ----
        if name == "weather":
            weather_map = {0.00: "Clear", 0.33: "Cloud", 0.66: "Rain", 1.00: "Snow"}
            cond = weather_map.get(round(float(v), 2), f"GT={float(v):.2f}")
            title = f"Histogram for weather scalar under {cond.lower()} conditions"

        elif name == "illum":
            illum_map = {0.00: "Day", 0.50: "Dusk", 1.00: "Night"}
            cond = illum_map.get(round(float(v), 2), f"GT={float(v):.2f}")
            title = f"Histogram for illumination scalar under {cond.lower()} conditions"

        else:
            title = f"Histogram for {name} scalar (gt={v})"

        plt.title(title)

        _savefig(
            outdir / f"hist_{name}_{gt_col}_{str(v).replace('.', 'p')}.png"
        )



def plot_scatter_pred_vs_gt(df: pd.DataFrame, outdir: Path, pred_col: str, gt_col: str, name: str, max_points=200000):
    d = df[[pred_col, gt_col]].dropna()
    if d.empty:
        return
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)

    plt.figure()
    plt.scatter(d[gt_col].values, d[pred_col].values, s=6, alpha=0.25)
    plt.xlabel(gt_col)
    plt.ylabel(pred_col)
    plt.title(f"{name}: pred vs gt (scatter)")
    _savefig(outdir / f"scatter_{name}_pred_vs_gt.png")


# -------------------------
# Analisi “shortcut” per sequenza
# -------------------------
def sequence_level_analysis(df: pd.DataFrame, outdir: Path, pred_col: str, gt_col: str, name: str):
    # Richiede seq_name valorizzato
    if "seq_name" not in df.columns:
        return
    d = df[[pred_col, gt_col, "seq_name", "split", "epoch"]].dropna()
    d = d[d["seq_name"].astype(str) != "None"]
    if d.empty:
        return

    # 1) media per sequenza (agg. su tutte le epoche o per epoca)
    seq_stats = (
        d.groupby(["split", "seq_name"])
         .agg(
             gt_mean=(gt_col, "mean"),
             pred_mean=(pred_col, "mean"),
             pred_std=(pred_col, "std"),
             n=(pred_col, "size"),
         )
         .reset_index()
    )
    seq_stats["abs_err_seq"] = np.abs(seq_stats["pred_mean"] - seq_stats["gt_mean"])
    seq_stats.to_csv(outdir / f"seq_stats_{name}.csv", index=False)

    # Scatter per sequenza: pred_mean vs gt_mean
    plt.figure()
    plt.scatter(seq_stats["gt_mean"].values, seq_stats["pred_mean"].values, s=18, alpha=0.7)
    plt.xlabel("gt_mean (seq)")
    plt.ylabel("pred_mean (seq)")
    plt.title(f"{name}: per-sequence means (each dot = one seq)")
    _savefig(outdir / f"seq_scatter_{name}.png")

    # 2) Quanto varia tra sequenze dentro la stessa classe GT?
    # Idea: se dentro la stessa classe GT (es. rain=0.66) le sequenze finiscono con pred_mean molto diverse,
    # allora lo scalare sta “portando” anche informazione di scena/identità (o comunque variabili non-meteo).
    within = (
        seq_stats.groupby(["split", "gt_mean"])
        .agg(
            pred_mean_std_across_seqs=("pred_mean", "std"),
            pred_mean_mean_across_seqs=("pred_mean", "mean"),
            n_seqs=("seq_name", "nunique"),
        )
        .reset_index()
    )
    within.to_csv(outdir / f"seq_withinclass_dispersion_{name}.csv", index=False)

    # Plot: dispersione tra sequenze per classe
    for split, g in within.groupby("split"):
        plt.figure()
        plt.bar([str(x) for x in g["gt_mean"].values], g["pred_mean_std_across_seqs"].values)
        plt.xlabel("gt_mean (class target)")
        plt.ylabel("std of pred_mean across sequences")
        plt.title(f"{name}: between-sequence dispersion within class ({split})")
        _savefig(outdir / f"seq_withinclass_dispersion_{name}_{split}.png")

    # 3) “ANOVA-like”: quanta varianza totale è spiegata dall’identità sequenza (dentro una classe)?
    # Calcoliamo, per ogni classe e split, una quota: var( pred_mean per seq ) / var( pred per sample )
    # Se questa quota è alta, l’ID sequenza spiega molto della variabilità del pred.
    ratios = []
    for (split, cls), g_cls in d.groupby(["split", gt_col]):
        # var sample-level
        var_sample = np.var(g_cls[pred_col].values.astype(float))
        # var seq-level means
        means = g_cls.groupby("seq_name")[pred_col].mean().values.astype(float)
        var_seqmean = np.var(means) if len(means) >= 2 else np.nan
        ratio = (var_seqmean / var_sample) if (np.isfinite(var_seqmean) and var_sample > 1e-12) else np.nan
        ratios.append({"split": split, "gt_class": float(cls), "var_ratio_seqmean_over_sample": ratio, "n_seqs": int(g_cls["seq_name"].nunique())})
    ratios = pd.DataFrame(ratios)
    ratios.to_csv(outdir / f"seq_identity_ratio_{name}.csv", index=False)

    for split, g in ratios.groupby("split"):
        plt.figure()
        plt.bar([str(x) for x in g["gt_class"].values], g["var_ratio_seqmean_over_sample"].values)
        plt.xlabel("gt class")
        plt.ylabel("var(seq means) / var(samples)")
        plt.title(f"{name}: how much sequence identity explains pred variability ({split})")
        _savefig(outdir / f"seq_identity_ratio_{name}_{split}.png")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to env_scalars_all.csv (WeatherScalarLogger output)")
    ap.add_argument("--outdir", type=str, default="env_scalar_analysis", help="Output directory for plots/csv summaries")
    ap.add_argument("--only_latest_epoch", action="store_true", help="If set, analyze only the max epoch per split")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    df = pd.read_csv(csv_path)

    # Sanity columns
    needed = ["split", "epoch", "s_weather_pred", "s_illum_pred"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {csv_path}. Columns are: {list(df.columns)}")

    # Se hai GT, fai analisi supervisionata completa
    has_weather_gt = "s_weather_gt" in df.columns and df["s_weather_gt"].notna().any()
    has_illum_gt   = "s_illum_gt" in df.columns and df["s_illum_gt"].notna().any()

    # Opzione: usa solo epoca finale (per split)
    if args.only_latest_epoch:
        keep = []
        for split, g in df.groupby("split"):
            emax = int(np.nanmax(g["epoch"].values))
            keep.append(g[g["epoch"] == emax])
        df = pd.concat(keep, axis=0, ignore_index=True)

    # -----------------
    # Summary per split/epoch
    # -----------------
    rows = []
    for (split, epoch), g in df.groupby(["split", "epoch"]):
        r = {"split": split, "epoch": int(epoch), "n": int(len(g))}
        if has_weather_gt:
            r["mae_weather"]  = _mae(g["s_weather_gt"], g["s_weather_pred"])
            r["rmse_weather"] = _rmse(g["s_weather_gt"], g["s_weather_pred"])
            r["r2_weather"]   = _safe_r2(g["s_weather_gt"], g["s_weather_pred"])
            r["corr_weather"] = _safe_corr(g["s_weather_gt"], g["s_weather_pred"])
        if has_illum_gt:
            r["mae_illum"]  = _mae(g["s_illum_gt"], g["s_illum_pred"])
            r["rmse_illum"] = _rmse(g["s_illum_gt"], g["s_illum_pred"])
            r["r2_illum"]   = _safe_r2(g["s_illum_gt"], g["s_illum_pred"])
            r["corr_illum"] = _safe_corr(g["s_illum_gt"], g["s_illum_pred"])
        rows.append(r)

    summary = pd.DataFrame(rows).sort_values(["split", "epoch"])
    summary.to_csv(outdir / "summary_by_split_epoch.csv", index=False)

    # Curve metriche su epoche
    if has_weather_gt:
        plot_epoch_curves(summary, outdir, "weather")
    if has_illum_gt:
        plot_epoch_curves(summary, outdir, "illum")

    # -----------------
    # Analisi calibrazione + distribuzioni
    # -----------------
    if has_weather_gt:
        plot_calibration_by_gt(df, outdir, "s_weather_pred", "s_weather_gt", "weather")
        plot_hist_by_gt(df, outdir, "s_weather_pred", "s_weather_gt", "weather")
        plot_scatter_pred_vs_gt(df, outdir, "s_weather_pred", "s_weather_gt", "weather")

        # Shortcut/seq analysis (se possibile)
        sequence_level_analysis(df, outdir, "s_weather_pred", "s_weather_gt", "weather")

    if has_illum_gt:
        plot_calibration_by_gt(df, outdir, "s_illum_pred", "s_illum_gt", "illum")
        plot_hist_by_gt(df, outdir, "s_illum_pred", "s_illum_gt", "illum")
        plot_scatter_pred_vs_gt(df, outdir, "s_illum_pred", "s_illum_gt", "illum")

        sequence_level_analysis(df, outdir, "s_illum_pred", "s_illum_gt", "illum")

    # -----------------
    # Report testuale minimale
    # -----------------
    report_lines = []
    report_lines.append(f"Loaded: {csv_path}")
    report_lines.append(f"Rows: {len(df)}")
    report_lines.append(f"Has weather gt: {has_weather_gt}")
    report_lines.append(f"Has illum gt: {has_illum_gt}")
    report_lines.append("")
    report_lines.append("Top-level summary (last epoch per split):")
    for split, g in summary.groupby("split"):
        g = g.sort_values("epoch")
        last = g.iloc[-1].to_dict()
        report_lines.append(f"- {split} @ epoch {int(last['epoch'])}, n={int(last['n'])}: " +
                            ", ".join([f"{k}={last[k]:.4f}" for k in last.keys() if k.startswith(("mae_", "rmse_", "r2_", "corr_")) and np.isfinite(last[k])]))

    (outdir / "REPORT.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nSaved analysis outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
