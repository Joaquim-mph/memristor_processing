import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import numpy as np
from typing import List, Dict, Any, Tuple
from matplotlib import animation
from matplotlib.animation import PillowWriter
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D 



def filter_experiments(df, date_prefix, wavelength, vsd_end=None, info=None):
    query_parts = [f"`Laser wavelength` == {wavelength}"]
    if vsd_end is not None:
        query_parts.append(f"`VSD end` == {vsd_end}")
    if info is not None:
        query_parts.append(f"`Information` == '{info}'")

    query_str = " and ".join(query_parts)

    return (
        df
        .query(query_str)
        .loc[lambda d: d["data_key"].str.startswith(date_prefix)]
        .sort_values("Laser voltage")
        .reset_index(drop=True)
    )


def filter_raw_and_hyst(props_raw, props_hyst, date_prefix, wavelength, info=None, vsd_end=None):
    raw_filtered = filter_experiments(
        df=props_raw,
        date_prefix=date_prefix,
        wavelength=wavelength,
        vsd_end=vsd_end,
        info=info
    )
    hyst_filtered = filter_experiments(
        df=props_hyst,
        date_prefix=date_prefix,
        wavelength=wavelength,
        vsd_end=vsd_end,
        info=info
    )
    return raw_filtered, hyst_filtered



def process_and_plot_all(
    date_prefix, wavelength, info,
    props_raw, props_hyst, data_raw, data_hyst,
    vsd_end=None, show = True
    ):

    # Ensure the figures directory exists
    
    base_dir = os.path.join("figures", date_prefix)
    
    os.makedirs(base_dir, exist_ok=True)

    # === Plotting logic (unchanged) ===
    def plot_lines_from_filtered(data, data_map, x_col, y_col=None, y_col_getter=None,
                                 title="", y_label="", filename="", marker='o'):
        
    
        n = len(data)
        colors = cm.magma(np.linspace(0, 1, n))
        fig, ax = plt.subplots()
        for i, (_, row) in enumerate(data.iterrows()):
            df = data_map[row["data_key"]]()
            if y_col == "I (A)":
                df[y_col] *= 1e6  # Convert to μA

            y_column = y_col_getter(df) if y_col_getter else y_col
            x, y = df[[x_col, y_column]].values.T
            ax.plot(
                x, y,
                label=f'{row["Laser voltage"]:.1f} V',
                color=colors[i],
                marker=marker if marker else None
            )
        ax.set_xlabel("$\\rm V_{DS}$ (V)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        #ax.legend(title="Laser voltage", loc="best")
        fig.tight_layout()
        outpath = os.path.join(base_dir, filename)
        fig.savefig(outpath, dpi=300)
        print(f"Saved → {outpath}")
        if show:
            plt.show()
        else:
            plt.close(fig)


    # === Apply new unified filter ===
    raw_data, hyst_data = filter_raw_and_hyst(
        props_raw=props_raw,
        props_hyst=props_hyst,
        date_prefix=date_prefix,
        wavelength=wavelength,
        info=info,
        vsd_end=vsd_end
    )

    # === Plot raw IV ===
    plot_lines_from_filtered(
        raw_data,
        data_map=data_raw,
        x_col="Vsd (V)",
        y_col="I (A)",
        #title=f"Raw IV - {wavelength} nm",
        y_label="$\\rm I_{{DS}}$ ($\\rm \mu$A)",
        filename=f"IV_{info}_{wavelength}_up.png",
    )

    # === Plot hysteresis ===
    hyst_map = {k: lambda k=k: data_hyst[k]() for k in hyst_data["data_key"]}
    plot_lines_from_filtered(
        hyst_data,
        data_map=hyst_map,
        x_col="Vsd (V)",
        y_col_getter=lambda df: df.columns[1],
        #title=f"Hysteresis - {wavelength} nm",
        y_label="Measured Hysteresis [A]",
        filename=f"hysteresis_{info}_{wavelength}.png",
        marker=None,
    )
  
def batch_process_and_plot(
    experiments: List[Dict[str, Any]],
    props_raw,
    data_raw,
    show = False
):
    """
    Loop over a list of experiment‐spec dicts and call process_and_plot_all for each.
    """
    for exp in experiments:
        print(f"→ plotting {exp['info']} ({exp['date_prefix']}, {exp['wavelength']} nm)")
        process_and_plot_all(
            date_prefix = exp["date_prefix"],
            wavelength  = exp["wavelength"],
            info        = exp["info"],
            props_raw   = props_raw,
            props_hyst  = exp["props_hyst"],
            data_raw    = data_raw,
            data_hyst   = exp["data_hyst"],
            vsd_end     = exp.get("vsd_end", None),
            show=show
        )




def build_calibration_functions(props_df, data_dict, date_prefix, wavelengths):
    calibration = {}
    for wl in wavelengths:
        filtered = props_df.loc[
            (props_df["Procedure type"] == "LaserCalibration") &
            (props_df["Laser wavelength"] == wl) &
            (props_df["data_key"].str.startswith(date_prefix))
        ]
        if filtered.empty:
            continue

        key = filtered["data_key"].iloc[0]
        df = data_dict[key]() if callable(data_dict[key]) else data_dict[key]
        f = interp1d(df["VL (V)"], df["Power (W)"], kind="linear", fill_value="extrapolate")
        calibration[wl] = f
    return calibration


def apply_calibration(df, voltage_col, wl, calibration_fns, area_factor, new_col="LED power"):
    if wl not in calibration_fns:
        raise ValueError(f"No calibration function found for {wl} nm")
    f = calibration_fns[wl]
    df[new_col] = df[voltage_col].apply(f) * area_factor
    return df



def overlay_hyst_vs_power_multi(
    experiments: List[Dict[str, Any]],
    props_raw,
    data_raw,
    area_factor: float,
    voltage_col: str = "Laser voltage",
    hyst_col: str    = "max_positive_hysteresis",
    new_col: str     = "LED power",
    baseline: bool   = False,
    outdir: str      = ".",
    outfname: str    = "hyst_overlay.png"
):
    """
    Overplot max-positive hysteresis vs. power for multiple experiments,
    each with its own date_prefix, wavelength, info, etc.

    Parameters
    ----------
    baseline : bool
        If True, subtract the first hysteresis point y[0] from all y before plotting.
    """
    fig, ax = plt.subplots()

    for exp in experiments:
        dp         = exp["date_prefix"]
        wl         = exp["wavelength"]
        info       = exp["info"]
        props_hyst = exp["props_hyst"]
        data_hyst  = exp["data_hyst"]
        vsd_end    = exp.get("vsd_end", None)

        # build calibrator for this date & wl
        cals = build_calibration_functions(
            props_df    = props_raw,
            data_dict   = data_raw,
            date_prefix = dp,
            wavelengths = [wl]
        )
        if wl not in cals:
            print(f"[warn] no calibration for {wl} nm on {dp} → skipping")
            continue

        # filter
        _, hyst = filter_raw_and_hyst(
            props_raw   = props_raw,
            props_hyst  = props_hyst,
            date_prefix = dp,
            wavelength  = wl,
            info        = info,
            vsd_end     = vsd_end
        )
        if hyst.empty:
            print(f"[warn] no hyst data for {wl} nm on {dp} → skipping")
            continue

        # calibrate
        hyst_cal = apply_calibration(
            df              = hyst.copy(),
            voltage_col     = voltage_col,
            wl              = wl,
            calibration_fns = cals,
            area_factor     = area_factor,
            new_col         = new_col
        )

        # extract x & y
        x = hyst_cal[new_col].values
        y = hyst_cal[hyst_col].values

        # subtract baseline if requested
        if baseline and len(y) > 0:
            y = y - y[0]

        # plot
        ax.plot(
            x, y,
            "o-",
            label=f"{wl} nm"
        )

    ax.set_xlabel("Effective Power [W]")
    ylabel = "Δ Max Positive Hysteresis [A]" if baseline else "Max Positive Hysteresis [A]"
    ax.set_ylabel(ylabel)
    ax.set_title("Hysteresis vs. Power")
    ax.legend(title="Wavelength")
    fig.tight_layout()

    # save
    os.makedirs(outdir, exist_ok=True)
    fullpath = os.path.join(outdir, outfname)
    fig.savefig(fullpath, dpi=300)
    print(f"Saved overlay plot → {fullpath}")

    plt.show()

def find_bad_iv_experiments_in_folder(data_raw, folder_name):
    bad_keys = []
    X_up_ref = None
    X_down_ref = None

    for key, data in data_raw.items():
        if not key.startswith(folder_name):
            continue  # Skip entries outside the target folder

        if not isinstance(data, dict):
            print(f"[SKIP] {key} is not a dictionary (type={type(data)}).")
            bad_keys.append(key)
            continue

        x_up = data.get('x_up', None)
        x_down = data.get('x_down', None)

        if x_up is None or x_down is None or len(x_up) == 0 or len(x_down) == 0:
            print(f"[EMPTY] {key} has missing or empty up/down sweep.")
            bad_keys.append(key)
            continue

        if X_up_ref is None:
            X_up_ref = x_up
            X_down_ref = x_down
        else:
            if not np.allclose(X_up_ref, x_up):
                print(f"[MISMATCH] Up-sweep grid mismatch in {key}")
                bad_keys.append(key)
                continue
            if not np.allclose(X_down_ref, x_down):
                print(f"[MISMATCH] Down-sweep grid mismatch in {key}")
                bad_keys.append(key)
                continue

    return bad_keys


#Visualization of many plots




def plot_mean_std_iv_branches(
    date_prefix: str,
    wavelength: int,
    info: str,
    props_raw,
    data_raw: dict,
    step = 10,
    split = True,
    vsd_end: float = None,
    x_col: str = "Vsd (V)",
    y_col: str = "I (A)",
    xlabel: str = "$V_{DS}$ (V)",
    ylabel: str = "$I_{DS}$ (A)",
    title: str = None,
    outdir: str = "figures",
    basename: str = "IV_mean_std",
    show: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Compute & plot pointwise mean±std of I vs Vsd for both
    forward (up-sweep) and backward (down-sweep) branches.

    Returns (fig_forward, fig_backward).
    """
    # 1) filter metadata
    meta = filter_experiments(
        df          = props_raw,
        date_prefix = date_prefix,
        wavelength  = wavelength,
        vsd_end     = vsd_end,
        info        = info
    )
    if meta.empty:
        raise RuntimeError(f"No IV runs for {info} on {date_prefix}")

    # 2) collect up/down data
    X_up = X_down = None
    Ys_up = []
    Ys_down = []

    for key in meta["data_key"]:
        # load the run
        df = data_raw[key]() if callable(data_raw[key]) else data_raw[key]

        # split into forward/backward by checking ΔVsd
        dv = df[x_col].diff().fillna(0)
        df_up   = df.loc[dv >= 0].reset_index(drop=True)
        df_down = df.loc[dv <  0].reset_index(drop=True)

        x_up,   y_up   = df_up[x_col].values,   df_up[y_col].values
        x_down, y_down = df_down[x_col].values, df_down[y_col].values

        # ensure common grids
        if X_up is None:
            X_up = x_up
        else:
            if not np.allclose(X_up, x_up):
                raise ValueError(f"Up-sweep grid mismatch in run {key}")
        if X_down is None:
            X_down = x_down
        else:
            if not np.allclose(X_down, x_down):
                raise ValueError(f"Down-sweep grid mismatch in run {key}")

        Ys_up.append(y_up)
        Ys_down.append(y_down)

    # 3) stack & compute mean/std
    Ymat_up   = np.vstack(Ys_up)
    Ymat_down = np.vstack(Ys_down)

    mu_up,   sigma_up   = Ymat_up.mean(axis=0),   Ymat_up.std(axis=0)
    mu_down, sigma_down = Ymat_down.mean(axis=0), Ymat_down.std(axis=0)

    # prepare output directory
    outdir = os.path.join(outdir, date_prefix)
    os.makedirs(outdir, exist_ok=True)


    if split:
        # 4a) plot forward
        fig_fwd, ax_fwd = plt.subplots()
        ax_fwd.errorbar(
            X_up[::step], mu_up[::step], yerr=sigma_up[::step],
            fmt='o', capsize=4, label="forward", elinewidth=1, ms=2
        )

        # 4b) plot backward
        fig_bwd, ax_bwd = plt.subplots()
        ax_bwd.errorbar(
            X_down[::step], mu_down[::step], yerr=sigma_down[::step],
            fmt='o', capsize=4, label="backward", elinewidth=1, ms=2
        )
        ax_fwd.set_xlabel(xlabel)
        ax_fwd.set_ylabel(ylabel)
        ax_fwd.set_title((title or info) + " (forward)")
        ax_fwd.legend()
        fig_fwd.tight_layout()
        fn_fwd = os.path.join(outdir, f"{basename}_forward.png")
        fig_fwd.savefig(fn_fwd, dpi=300)
        print(f"Saved → {fn_fwd}")


        ax_bwd.set_xlabel(xlabel)
        ax_bwd.set_ylabel(ylabel)
        ax_bwd.set_title((title or info) + " (backward)")
        ax_bwd.legend()
        fig_bwd.tight_layout()
        fn_bwd = os.path.join(outdir, f"{basename}_backward.png")
        fig_bwd.savefig(fn_bwd, dpi=300)
        print(f"Saved → {fn_bwd}")

    else:
        fig, ax = plt.subplots()
        ax.errorbar(
            X_up[::step], mu_up[::step], yerr=sigma_up[::step],
            fmt='o', capsize=4, label="forward", elinewidth=1, ms=2, errorevery=3
        )

        ax.errorbar(
            X_down[::step], mu_down[::step], yerr=sigma_down[::step],
            fmt='x', capsize=4, label="backward", elinewidth=1, ms=2, errorevery=3
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title((title or info) + " (forward and backward)")
        ax.legend()
        fig.tight_layout()
        fn = os.path.join(outdir, f"{basename}_forward_backward.png")
        fig.savefig(fn, dpi=300)
        print(f"Saved → {fn}")

    # show or close
    if show:
        plt.show()
    else:
        plt.close(fig_fwd)
        plt.close(fig_bwd)



def plot_iv_3d_surface(
    date_prefix: str,
    wavelength: int,
    info: str,
    props_raw,
    data_raw: Dict[str, Any],
    vsd_end: float     = None,
    x_col: str         = "Vsd (V)",
    y_col: str         = "I (A)",
    xlabel: str        = "$V_{DS}$ (V)",
    ylabel: str        = "$I_{DS}$ (A)",
    title: str         = None,
    outdir: str        = "figures",
    basename: str      = "IV_3d",
    cmap: str          = "viridis",
    show: bool         = True
):
    """
    Build a 3D surface of all IV runs:

      - x-axis: Vsd (V)
      - y-axis: run index (0…N-1)
      - z-axis: I (A)
    """
    # 1) filter metadata
    meta = filter_experiments(
        df          = props_raw,
        date_prefix = date_prefix,
        wavelength  = wavelength,
        vsd_end     = vsd_end,
        info        = info
    )
    if meta.empty:
        raise RuntimeError(f"No IV runs for {info} on {date_prefix}")

    # 2) load each run into a list
    X = None
    Ys = []
    for key in meta["data_key"]:
        df = data_raw[key]() if callable(data_raw[key]) else data_raw[key]
        x = df[x_col].values
        y = df[y_col].values

        # ensure a common X grid
        if X is None:
            X = x
        else:
            if not np.allclose(X, x):
                raise ValueError(f"Vsd grid mismatch in run {key}")

        Ys.append(y)

    Ymat = np.vstack(Ys)             # shape = (n_runs, n_points)
    runs, pts = Ymat.shape

    # 3) build meshgrid for plotting
    idx = np.arange(runs)
    R, Xmesh = np.meshgrid(idx, X, indexing='ij')  # both shape (runs, pts)
    Z = Ymat

    # 4) make output dir
    folder = os.path.join(outdir, date_prefix)
    os.makedirs(folder, exist_ok=True)

    # 5) plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        Xmesh, R, Z,
        cmap=cmap,
        edgecolor='none',
        alpha=0.8
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=ylabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Run index")
    ax.set_zlabel(ylabel)
    ax.set_title((title or info) + f"\n3D surface ({runs} runs)")

    fig.tight_layout()
    outpath = os.path.join(folder, f"{basename}_surface.png")
    fig.savefig(outpath, dpi=300)
    print(f"Saved → {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



def plot_iv_3d_surface_interactive(
    date_prefix: str,
    wavelength: int,
    info: str,
    props_raw,
    data_raw: Dict[str, Any],
    vsd_end: float     = None,
    x_col: str         = "Vsd (V)",
    y_col: str         = "I (A)",
    outdir: str        = "figures",
    basename: str      = "IV_surface_interactive"
):
    # 1) filter metadata
    meta = filter_experiments(
        df          = props_raw,
        date_prefix = date_prefix,
        wavelength  = wavelength,
        vsd_end     = vsd_end,
        info        = info
    )
    if meta.empty:
        raise RuntimeError(f"No IV runs for {info} on {date_prefix}")

    # 2) load each run
    X = None
    Ys = []
    for key in meta["data_key"]:
        df = data_raw[key]() if callable(data_raw[key]) else data_raw[key]
        x = df[x_col].values
        y = df[y_col].values
        if X is None:
            X = x
        else:
            if not np.allclose(X, x):
                raise ValueError(f"Grid mismatch in run {key}")
        Ys.append(y)

    Ymat = np.vstack(Ys)
    runs, pts = Ymat.shape

    # 3) build mesh
    idx = np.arange(runs)
    R, Xmesh = np.meshgrid(idx, X, indexing='ij')

    # 4) plotly surface colored by run index
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x            = Xmesh,
        y            = R,
        z            = Ymat,
        surfacecolor = R,
        cmin         = R.min(),      # <-- lock the color‐mapping to [0, runs-1]
        cmax         = R.max(),
        colorscale   = "Viridis",
        colorbar     = dict(title="Run index"),
        showscale    = True
    ))
    fig.update_layout(
        title=f"IV surface: {info}",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title="Run index",
            zaxis_title=y_col
        ),
        autosize=True,
    )

    # 5) save and show
    folder = os.path.join(outdir, date_prefix)
    os.makedirs(folder, exist_ok=True)
    html_path = os.path.join(folder, f"{basename}.html")
    fig.write_html(html_path)
    print(f"Saved interactive surface → {html_path}")


def create_iv_gif_fixed_axes(
    date_prefix: str,
    wavelength: int,
    info: str,
    props_raw,
    data_raw: Dict[str, Any],
    outdir: str = "figures",
    basename: str = "iv_sequence",
    x_col: str = "Vsd (V)",
    y_col: str = "I (A)",
    interval: int = 100,          # ms between frames
    dpi: int = 200
):
    """
    Generate a GIF of each IV run sequentially with fixed x/y limits across all frames.
    """
    # 1) filter metadata for the IV runs
    meta = filter_experiments(
        df          = props_raw,
        date_prefix = date_prefix,
        wavelength  = wavelength,
        info        = info
    )
    if meta.empty:
        raise RuntimeError(f"No IV runs for {info} on {date_prefix}")

    # 2) gather all data to compute global limits
    all_y = []
    all_x = []
    for key in meta["data_key"]:
        df = data_raw[key]() if callable(data_raw[key]) else data_raw[key]
        all_y.append(df[y_col].values)
        all_x.append(df[x_col].values)
    all_y = np.concatenate(all_y)
    all_x = np.concatenate(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_min, x_max = np.min(all_x), np.max(all_x)

    # 3) prepare output directory
    folder = os.path.join(outdir, date_prefix)
    os.makedirs(folder, exist_ok=True)
    gif_path = os.path.join(folder, f"{basename}.gif")

    # 4) set up the figure and axis with fixed limits
    fig, ax = plt.subplots()
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    title_text = ax.set_title(f"[key]")


    def update(frame_idx):
        key = meta["data_key"].iloc[frame_idx]
        df  = data_raw[key]() if callable(data_raw[key]) else data_raw[key]
        x   = df[x_col].values
        y   = df[y_col].values

        ax.clear()
        ax.set_xlim(x_min-0.5, x_max+0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.plot(x, y, '-', markersize=2, linewidth=1)

        # here’s the important bit: set the title each frame
        ax.set_title(
            f"{info}\n"
            f"Run {frame_idx+1}/{len(meta)}"
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(meta),
        interval=interval, repeat_delay=1000
    )

    writer = PillowWriter(fps=1000/interval)
    anim.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved IV animation → {gif_path}")

