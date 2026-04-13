"""
Shared matplotlib style configuration for memristor plots.

Matches the notebook style: scienceplots + nature theme, no grids,
publication-quality defaults.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["nature", "science"])
except ImportError:
    pass


def apply_default_style():
    """Apply the default plot style used across all scripts."""
    mpl.rcParams.update({
        "lines.linewidth": 2,
        "lines.markersize": 4,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "axes.linewidth": 2.5,
        "legend.fontsize": 15,
        "legend.handlelength": 3,
        "legend.handleheight": 1,
        "legend.handletextpad": 2,
        "legend.labelspacing": 0.6,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.figsize": (10, 10),
        "axes.prop_cycle": plt.cycler(color=[
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
            "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
            "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
            "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
        ]),
    })
