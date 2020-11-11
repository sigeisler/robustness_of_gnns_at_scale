import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Callable, Tuple, Union
from threading import Timer
import numpy as np
import matplotlib as mpl
from IPython.display import IFrame
import pylab

is_production = False


def enable_production_mode():
    global is_production
    mpl.use('pgf')
    is_production = True


def dedicated_legend_plot(filename: str,
                          ax: mpl.axes.Axes = None,
                          format: str = 'pgf',
                          dpi: float = 600,
                          mod_label: Callable[[str], str] = None,
                          **kwargs):
    global is_production
    if not is_production:
        return
    if ax is None:
        ax = plt.gca()
    if os.path.splitext(filename)[1] == format:
        filepath = filename
    else:
        filepath = f"{filename}.{format}"
    figLegend = pylab.figure()
    handles, labels = ax.get_legend_handles_labels()
    if mod_label is not None:
        labels = [mod_label(l) for l in labels]
    pylab.figlegend(handles, labels, loc='center', **kwargs)
    figLegend.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def figsize(scale: float,
            ratio_yx: Union[float, Tuple[float, float]] = None,
            textwidth_pt: float = 397.48499) -> Tuple[float, float]:
    """Get an appropriate figure size.

    Parameters
    ----------
    scale
        Size of the figure relative to the text width.
    ratio_yx
        Ratio of height by width.
    textwidth_pt
        Text width in the Latex file. Get this from LaTeX using \\the\\textwidth.

    Returns
    -------
    float
        Figure width
    float
        Figure height
    """
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    if ratio_yx is None:
        golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio
        ratio_yx = golden_mean
    if isinstance(ratio_yx, tuple):
        ratio_yx = ratio_yx[1] / ratio_yx[0]

    global is_production
    if not is_production:
        x = 12
        y = x * ratio_yx
        return (x, y)

    fig_width = textwidth_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio_yx  # height in inches
    return fig_width, fig_height


def sns_facetsize(
        tot_width: float = 0.95, ratio_yx_facet: float = 1.6,
        nrows: int = 1, ncols: int = 1,
        textwidth_pt: float = 397.48499) -> Tuple[float, float]:
    ratio_yx = ratio_yx_facet * nrows / ncols
    size = figsize(tot_width, ratio_yx, textwidth_pt)
    height_facet = size[1] / nrows
    ratio_xy_facet = 1 / ratio_yx_facet
    return height_facet, ratio_xy_facet


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "text.latex.preamble": [
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\newcommand*{\mat}[1]{\boldsymbol{#1}}",
    ],
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\newcommand*{\mat}[1]{\boldsymbol{#1}}",
    ],
}
mpl.rcParams.update(pgf_with_latex)


# Set line widths
default_rcParams = {
    'grid.linewidth': 0.5,
    'grid.color': '.8',
    'axes.linewidth': 0.75,
    'axes.edgecolor': '.7',
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.minor.width': 0.75,
    'ytick.minor.width': 0.75,
    'xtick.minor.size': 2.0,
    'ytick.minor.size': 2.0,
    'lines.linewidth': 1.0,
}


def set_style(style: str = 'whitegrid', rcParams: dict = {}):
    if style:
        sns.set(style=style, color_codes=True)  # , palette='colorblind', color_codes=True)
    else:
        sns.set(palette='colorblind', color_codes=True)
    mpl.rcParams.update(pgf_with_latex)
    mpl.rcParams.update(default_rcParams)
    mpl.rcParams.update(rcParams)


# Customized newfig and savefig functions
def newfig(
        width: float,
        ratio_yx: float = None,
        style: str = 'whitegrid',
        rcParams: dict = {},
        nrows: int = 1,
        ncols: int = 1,
        textwidth_pt: float = 397.48499,
        **subplots_kws) -> Tuple[mpl.figure.Figure, "np.ndarray[mpl.axes._subplots.AxesSubplot]"]:
    # plt.clf()
    set_style(style=style, rcParams=rcParams)
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize(width, ratio_yx=ratio_yx,
                        textwidth_pt=textwidth_pt),
        **subplots_kws
    )


def savefig(
        filename: str, fig: mpl.figure.Figure = None,
        dpi: float = 600, format: str = 'pgf', preview: str = 'pdf',
        close_fig: bool = True, remove_preview_file_after: float = 10, **kwargs) -> Union[IFrame, None]:
    global is_production
    if not is_production:
        from IPython import display
        display.display(fig)
        return
    if fig is None:
        fig = plt.gca().figure
    if os.path.splitext(filename)[1] == format:
        filepath = filename
    else:
        filepath = f"{filename}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
    if close_fig:
        if fig is None:
            plt.close()
        else:
            plt.close(fig)
    if preview is not None:
        if preview == format:
            preview_path = filepath
        else:
            while True:
                rnd_int = np.random.randint(np.iinfo(np.uint32).max)
                preview_path = f"preview_tmp{rnd_int}.{preview}"
                if not os.path.exists(preview_path):
                    break
            fig.savefig(preview_path, dpi=dpi, bbox_inches='tight', **kwargs)
            Timer(remove_preview_file_after, os.remove, args=[preview_path]).start()
        return IFrame(os.path.relpath(preview_path), width=700, height=500)


if __name__ == "__main__":
    # Simple plot
    fig, ax = newfig(0.6)

    def ema(y, a):
        s = []
        s.append(y[0])
        for t in range(1, len(y)):
            s.append(a * y[t] + (1 - a) * s[t - 1])
        return np.array(s)

    y = [0] * 200
    y.extend([20] * (1000 - len(y)))
    s = ema(y, 0.01)

    ax.plot(s)
    ax.set_xlabel('X Label')
    ax.set_ylabel('EMA')
