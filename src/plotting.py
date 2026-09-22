# src/plotting.py
"""
Shared figure-saving helper so every notebook writes its plots to outputs/
in a consistent place, instead of only displaying them inline.
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def save_fig(fig, name: str, notebook: str, dpi: int = 150, also_show: bool = True):
    """
    Save a matplotlib figure to outputs/<notebook>/<name>.png, then optionally
    display it inline as usual.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save (from `fig, ax = plt.subplots(...)`).
    name : str
        Filename without extension, e.g. "target_distribution".
    notebook : str
        Subfolder to save under, e.g. "02_EDA" -- keep this the same for
        every plot in a given notebook so its outputs land in one folder.
    dpi : int
        Resolution to save at (150 is fine for slides/reports; use 300 for print).
    also_show : bool
        If True, calls plt.show() after saving (so notebook output is unchanged).
    """
    out_dir = OUTPUTS_ROOT / notebook
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if also_show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
