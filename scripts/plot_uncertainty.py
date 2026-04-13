#!/usr/bin/env python3
# plot_uncertainty.py
# Usage:
#   python plot_uncertainty.py --npz_dir /path/to/out_dir --out_dir /path/to/plots
#   python plot_uncertainty.py --npz /path/to/uncertainty_val_ep010.npz

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Simple rankdata (average ranks for ties), no scipy dependency.
    Returns ranks starting at 1.
    """
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

    # Handle ties: average the ranks for equal values
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom <= 1e-12:
        return np.nan
    return float((x * y).sum() / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


def quantile_bins(x: np.ndarray, n_bins: int = 10):
    """
    Returns bin edges using quantiles, robust to heavy-tailed distributions.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)
    # Ensure strictly increasing edges (avoid duplicates in degenerate cases)
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback: uniform bins
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    return edges


def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """
    Bin by x (quantile edges) and compute mean(y) per bin + count.
    Returns:
      x_mid, y_mean, counts
    """
    edges = quantile_bins(x, n_bins=n_bins)
    # digitize: bins are [edges[k], edges[k+1]) except last
    idx = np.digitize(x, edges[1:-1], right=False)
    x_mid = []
    y_mean = []
    counts = []

    for b in range(len(edges) - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        xb = x[mask]
        yb = y[mask]
        x_mid.append(np.median(xb))
        y_mean.append(np.mean(yb))
        counts.append(int(mask.sum()))

    return np.array(x_mid), np.array(y_mean), np.array(counts)


def load_npz(npz_path: str):
    d = np.load(npz_path)
    # expected keys (from your training logger)
    logvar_t = d["logvar_t"].astype(np.float64)
    logvar_q = d["logvar_q"].astype(np.float64)
    trans_err = d["trans_err_m"].astype(np.float64)
    rot_err_deg = d["rot_err_deg"].astype(np.float64)
    return logvar_t, logvar_q, trans_err, rot_err_deg


def make_plots(npz_path: str, out_dir: str, n_bins: int = 10):
    os.makedirs(out_dir, exist_ok=True)

    logvar_t, logvar_q, trans_err, rot_err_deg = load_npz(npz_path)

    # Convert log-variance to sigma (standard deviation)
    sigma_t = np.exp(0.5 * logvar_t)
    sigma_q = np.exp(0.5 * logvar_q)

    # Correlations
    sp_t = spearmanr(sigma_t, trans_err)
    pr_t = pearsonr(sigma_t, trans_err)
    sp_q = spearmanr(sigma_q, rot_err_deg)
    pr_q = pearsonr(sigma_q, rot_err_deg)

    base = os.path.splitext(os.path.basename(npz_path))[0]

    # --------------------------
    # 1) Scatter: sigma vs error
    # --------------------------
    plt.figure()
    plt.scatter(sigma_t, trans_err, s=8, alpha=0.35)
    plt.xscale("log")
    plt.xlabel("Predicted σ_t  (exp(0.5*logvar_t))  [log scale]")
    plt.ylabel("Translation error ||t_pred - t_gt||2  [m]")
    plt.title(f"{base} | σ_t vs trans error | Spearman={sp_t:.3f}, Pearson={pr_t:.3f}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_scatter_sigma_t_vs_trans_err.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(sigma_q, rot_err_deg, s=8, alpha=0.35)
    plt.xscale("log")
    plt.xlabel("Predicted σ_q  (exp(0.5*logvar_q))  [log scale]")
    plt.ylabel("Rotation error (angle)  [deg]")
    plt.title(f"{base} | σ_q vs rot error | Spearman={sp_q:.3f}, Pearson={pr_q:.3f}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_scatter_sigma_q_vs_rot_err.png"), dpi=200)
    plt.close()

    # ----------------------------------------
    # 2) Reliability-ish: bin by sigma, mean err
    # ----------------------------------------
    xmid_t, ymean_t, cnt_t = binned_stats(sigma_t, trans_err, n_bins=n_bins)
    plt.figure()
    plt.plot(xmid_t, ymean_t, marker="o")
    plt.xscale("log")
    plt.xlabel("Predicted σ_t (bin median)  [log scale]")
    plt.ylabel("Mean translation error in bin  [m]")
    plt.title(f"{base} | Binned: mean trans error vs σ_t (n_bins={n_bins})")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_binned_sigma_t_vs_trans_err.png"), dpi=200)
    plt.close()

    xmid_q, ymean_q, cnt_q = binned_stats(sigma_q, rot_err_deg, n_bins=n_bins)
    plt.figure()
    plt.plot(xmid_q, ymean_q, marker="o")
    plt.xscale("log")
    plt.xlabel("Predicted σ_q (bin median)  [log scale]")
    plt.ylabel("Mean rotation error in bin  [deg]")
    plt.title(f"{base} | Binned: mean rot error vs σ_q (n_bins={n_bins})")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_binned_sigma_q_vs_rot_err.png"), dpi=200)
    plt.close()

    # --------------------------
    # 3) Print a short summary
    # --------------------------
    summary_path = os.path.join(out_dir, f"{base}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"File: {npz_path}\n")
        f.write(f"N points: {len(trans_err)}\n\n")
        f.write("Translation:\n")
        f.write(f"  Spearman(σ_t, trans_err) = {sp_t:.6f}\n")
        f.write(f"  Pearson (σ_t, trans_err) = {pr_t:.6f}\n\n")
        f.write("Rotation:\n")
        f.write(f"  Spearman(σ_q, rot_err_deg) = {sp_q:.6f}\n")
        f.write(f"  Pearson (σ_q, rot_err_deg) = {pr_q:.6f}\n")

    print(f"[OK] Saved plots + summary to: {out_dir}")
    print(f"  Translation: Spearman={sp_t:.3f} Pearson={pr_t:.3f}")
    print(f"  Rotation   : Spearman={sp_q:.3f} Pearson={pr_q:.3f}")


def find_latest_npz(npz_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(npz_dir, "uncertainty_val_ep*.npz")))
    if not files:
        raise FileNotFoundError(f"No uncertainty_val_ep*.npz found in: {npz_dir}")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None, help="Path to a single .npz file.")
    ap.add_argument("--npz_dir", type=str, default=None, help="Directory containing uncertainty_val_ep*.npz")
    ap.add_argument("--out_dir", type=str, default="uncertainty_plots", help="Output directory for plots.")
    ap.add_argument("--bins", type=int, default=10, help="Number of quantile bins for binned plots.")
    args = ap.parse_args()

    if args.npz is None and args.npz_dir is None:
        raise SystemExit("Provide either --npz FILE or --npz_dir DIR")

    npz_path = args.npz if args.npz is not None else find_latest_npz(args.npz_dir)
    make_plots(npz_path, args.out_dir, n_bins=args.bins)


if __name__ == "__main__":
    main()
