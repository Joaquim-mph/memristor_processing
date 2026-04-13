"""
Plot IV sweep statistics: mean +/- std for forward and backward branches.

Reproduces the plots from no_light_statistic.ipynb:
  1. Mean ± std IV for forward and backward branches (split or combined)
  2. 3D surface of IV runs (static matplotlib)

Usage:
    python scripts/plot_iv_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_data import load_all_raw, filter_experiments
from scripts.plot_style import apply_default_style

apply_default_style()

OUTDIR = Path("figures")


def _split_sweep(df, x_col="Vsd (V)", y_col="I (A)"):
    """Split IV data into forward (dV >= 0) and backward (dV < 0) branches."""
    v = df[x_col].to_numpy()
    i = df[y_col].to_numpy()
    dv = np.diff(v, prepend=v[0])

    fwd_mask = dv >= 0
    bwd_mask = dv < 0

    return (v[fwd_mask], i[fwd_mask]), (v[bwd_mask], i[bwd_mask])


def plot_mean_std_iv(
    props,
    data,
    *,
    step: int = 10,
    split: bool = True,
    outdir: Path = OUTDIR,
    basename: str = "IV_mean_std",
    show: bool = False,
):
    """
    Compute and plot pointwise mean ± std of IV for forward/backward branches.

    Parameters
    ----------
    step : int
        Plot every N-th point (for readability with error bars).
    split : bool
        If True, separate forward/backward into two figures.
        If False, overlay on one figure.
    """
    n = props.height
    if n == 0:
        print("  [skip] No experiments")
        return

    # Collect forward/backward arrays
    X_fwd = X_bwd = None
    Ys_fwd, Ys_bwd = [], []

    for i in range(n):
        row = props.row(i, named=True)
        key = row["data_key"]
        if key not in data:
            continue

        df = data[key]
        (x_f, y_f), (x_b, y_b) = _split_sweep(df)

        # Ensure common grid
        if X_fwd is None:
            X_fwd, X_bwd = x_f, x_b
        else:
            if len(x_f) != len(X_fwd) or not np.allclose(X_fwd, x_f, atol=0.02):
                continue
            if len(x_b) != len(X_bwd) or not np.allclose(X_bwd, x_b, atol=0.02):
                continue

        Ys_fwd.append(y_f)
        Ys_bwd.append(y_b)

    if not Ys_fwd:
        print("  [skip] No valid sweeps with matching grids")
        return

    Ymat_fwd = np.vstack(Ys_fwd)
    Ymat_bwd = np.vstack(Ys_bwd)

    mu_fwd, sigma_fwd = Ymat_fwd.mean(axis=0), Ymat_fwd.std(axis=0)
    mu_bwd, sigma_bwd = Ymat_bwd.mean(axis=0), Ymat_bwd.std(axis=0)

    print(f"  Averaged {len(Ys_fwd)} sweeps")

    outdir.mkdir(parents=True, exist_ok=True)

    if split:
        # Forward
        fig_f, ax_f = plt.subplots()
        ax_f.errorbar(
            X_fwd[::step], mu_fwd[::step], yerr=sigma_fwd[::step],
            fmt="o", capsize=4, label="forward", elinewidth=1, ms=2,
        )
        ax_f.set_xlabel(r"$V_{DS}$ (V)")
        ax_f.set_ylabel(r"$I_{DS}$ (A)")
        ax_f.legend()
        fig_f.tight_layout()
        fpath = outdir / f"{basename}_forward.png"
        fig_f.savefig(fpath, dpi=300)
        print(f"  Saved -> {fpath}")

        # Backward
        fig_b, ax_b = plt.subplots()
        ax_b.errorbar(
            X_bwd[::step], mu_bwd[::step], yerr=sigma_bwd[::step],
            fmt="o", capsize=4, label="backward", elinewidth=1, ms=2,
        )
        ax_b.set_xlabel(r"$V_{DS}$ (V)")
        ax_b.set_ylabel(r"$I_{DS}$ (A)")
        ax_b.legend()
        fig_b.tight_layout()
        bpath = outdir / f"{basename}_backward.png"
        fig_b.savefig(bpath, dpi=300)
        print(f"  Saved -> {bpath}")

        if show:
            plt.show()
        else:
            plt.close(fig_f)
            plt.close(fig_b)
    else:
        fig, ax = plt.subplots()
        ax.errorbar(
            X_fwd[::step], mu_fwd[::step], yerr=sigma_fwd[::step],
            fmt="o", capsize=4, label="forward", elinewidth=1, ms=2, errorevery=3,
        )
        ax.errorbar(
            X_bwd[::step], mu_bwd[::step], yerr=sigma_bwd[::step],
            fmt="x", capsize=4, label="backward", elinewidth=1, ms=2, errorevery=3,
        )
        ax.set_xlabel(r"$V_{DS}$ (V)")
        ax.set_ylabel(r"$I_{DS}$ (A)")
        ax.legend()
        fig.tight_layout()
        fpath = outdir / f"{basename}_forward_backward.png"
        fig.savefig(fpath, dpi=300)
        print(f"  Saved -> {fpath}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_iv_3d_surface(
    props,
    data,
    *,
    outdir: Path = OUTDIR,
    basename: str = "IV_3d",
    cmap: str = "viridis",
    show: bool = False,
):
    """Build a 3D surface of all IV runs: x=Vsd, y=run index, z=I."""
    n = props.height
    if n == 0:
        return

    X = None
    Ys = []

    for i in range(n):
        row = props.row(i, named=True)
        key = row["data_key"]
        if key not in data:
            continue

        df = data[key]
        x = df["Vsd (V)"].to_numpy()
        y = df["I (A)"].to_numpy()

        if X is None:
            X = x
        elif len(x) != len(X) or not np.allclose(X, x, atol=0.02):
            continue

        Ys.append(y)

    if not Ys:
        print("  [skip] No valid sweeps with matching grids")
        return

    Ymat = np.vstack(Ys)
    runs, pts = Ymat.shape

    idx = np.arange(runs)
    R, Xmesh = np.meshgrid(idx, X, indexing="ij")

    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xmesh, R, Ymat, cmap=cmap, edgecolor="none", alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r"$I_{DS}$ (A)")
    ax.set_xlabel(r"$V_{DS}$ (V)")
    ax.set_ylabel("Run index")
    ax.set_zlabel(r"$I_{DS}$ (A)")
    ax.set_title(f"IV surface ({runs} runs)")
    fig.tight_layout()

    fpath = outdir / f"{basename}_surface.png"
    fig.savefig(fpath, dpi=300)
    print(f"  Saved -> {fpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Experiment definitions (from no_light_statistic.ipynb) ──

EXPERIMENTS = [
    {
        "folder_prefix": "2025-06-26",
        "information": "SnS - no light - PAIN",
        "wavelength": 385.0,
        "vsd_end": 8.0,
        "step": 10,
        "label": "PAIN_no_light",
    },
    {
        "folder_prefix": "2025-05-06",
        "information": "SnS - Stability",
        "wavelength": 0.0,
        "vsd_end": 9.0,
        "step": 1,
        "label": "stability_100um",
    },
]


def main():
    print("Loading all raw data...")
    props, data = load_all_raw()
    print(f"Loaded {props.height} experiments\n")

    for exp in EXPERIMENTS:
        label = exp["label"]
        filtered = filter_experiments(
            props,
            folder_prefix=exp.get("folder_prefix"),
            information=exp.get("information"),
            wavelength=exp.get("wavelength"),
            vsd_end=exp.get("vsd_end"),
        )
        print(f"\n=== [{label}] {filtered.height} sweeps ===")

        folder = exp.get("folder_prefix", "misc")
        step = exp.get("step", 10)

        # Mean ± std (split)
        print("Mean ± std (split):")
        plot_mean_std_iv(
            filtered, data,
            step=step,
            split=True,
            outdir=OUTDIR / folder,
            basename=f"IV_{label}",
        )

        # Mean ± std (combined)
        print("Mean ± std (combined):")
        plot_mean_std_iv(
            filtered, data,
            step=step,
            split=False,
            outdir=OUTDIR / folder,
            basename=f"IV_{label}",
        )

        # 3D surface
        print("3D surface:")
        plot_iv_3d_surface(
            filtered, data,
            outdir=OUTDIR / folder,
            basename=f"IV_{label}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
