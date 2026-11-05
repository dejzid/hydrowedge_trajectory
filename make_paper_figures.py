"""Paper figures, built only from stored case results (results/cases/<name>/).

Missing cases are run first (run_trajectories.process). Figures go to results/paper/;
an existing figure is kept unless --force is given (the case results themselves are
never touched).

  fig_axial_{pusher,puller}_s1         axial dipole: alpha = pi/3, sigma/(v0 l^2) = +-1,
                                       rho0 = 10, beta0 = 0, +-pi/12, +-pi/6, +-pi/4
  fig_transverse_{pusher,puller}_s10   transverse dipole: alpha = pi/3, sigma/(v0 l^2) = +-10,
                                       rho0 = 10, beta0 = pi/6, 8 headings theta = beta + delta

The panels (PNG, 600 dpi, typeset with LaTeX) are 3.375 in wide (one REVTeX column) with
the same axes width and scale; include them at natural size (or all at the same width)
so that the fonts match. fig_axial and fig_transverse can also draw other cases (e.g.
with wall repulsion) under a file-name suffix.

Usage:  python make_paper_figures.py [--force]
"""

import argparse
import csv
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_trajectories as runner  # noqa: E402

OUT = runner.ROOT / "results" / "paper"
STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
}
TEX = {"text.usetex": True, "font.serif": ["Computer Modern Roman"]}


def case(name):
    """Stored results of a case (run first if missing)."""
    out = runner.RESULTS / name
    if not (out / "case.json").exists():
        runner.process(runner.CASES / f"{name}.json")
    meta = json.loads((out / "case.json").read_text(encoding="utf-8"))
    return out, {**runner.DEFAULTS, **meta["input"]}


def read_csv(path):
    with open(path, newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


def paths(name):
    """{(sigma, start): dict of arrays} from trajectories.csv, and the case parameters."""
    out, p = case(name)
    runs = defaultdict(lambda: defaultdict(list))
    for r in read_csv(out / "trajectories.csv"):
        key = (float(r["sigma"]), int(r["start"]))
        for k in ("t", "x", "y", "beta", "delta", "z", "wall_distance"):
            runs[key][k].append(float(r[k]))
    ends = {(float(r["sigma"]), int(r["start"])): r["end"] for r in read_csv(out / "summary.csv")}
    return {k: {**{c: np.array(v) for c, v in d.items()}, "end": ends[k]} for k, d in runs.items()}, p


def alpha_of(p):
    return float(runner.angle(p["alpha"]))


def draw_wedge(ax, alpha, r):
    ax.fill([0, r * np.cos(alpha), r * np.cos(alpha), -r, -r],
            [0, r * np.sin(alpha), r * 2, r * 2, 0], color="0.88", lw=0)
    ax.fill([0, r * np.cos(alpha), r * np.cos(alpha), -r, -r],
            [0, -r * np.sin(alpha), -r * 2, -r * 2, 0], color="0.88", lw=0)
    ax.plot([r * np.cos(alpha), 0, r * np.cos(alpha)], [r * np.sin(alpha), 0, -r * np.sin(alpha)],
            "k", lw=1.2)


def draw_paths(ax, runs, sigma, alpha, view, labels=None, xlim=None, ylim=None, ends=True):
    keys = sorted(k for k in runs if k[0] == sigma)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
    xlim = xlim or (-0.02 * view, view)
    ylim = ylim or (-view * np.sin(alpha) * 1.02, view * np.sin(alpha) * 1.02)
    draw_wedge(ax, alpha, view * 1.5)
    for n, (c, k) in enumerate(zip(colors, keys)):
        r = runs[k]
        ax.plot(r["x"], r["y"], color=c, label=labels[n] if labels else None)
        ax.plot(r["x"][0], r["y"][0], "o", color=c, ms=3)
        if ends and r["end"] == "wall":
            ax.plot(r["x"][-1], r["y"][-1], "x", color=c, ms=5, mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$x/\ell$")
    ax.set_ylabel(r"$y/\ell$")


def save(fig, name, force, exts=("pdf", "png"), dpi=300):
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f"{name}.{ext}" for ext in exts]
    if all(t.exists() for t in targets) and not force:
        print(f"{name}: exists, kept (use --force to redraw)")
    else:
        for t in targets:
            fig.savefig(t, dpi=dpi)
        print(f"{name}: written to {OUT}")
    plt.close(fig)


# Single-panel layout, in inches: every figure is one REVTeX column wide with the same
# axes width and the same scale (inches per swimmer size), so fonts and lengths match
# between figures when they are included at their natural size; only the height
# follows the plotted y-range.
PANEL = dict(width=3.375, left=0.47, axes=1.85, bottom=0.40, top=0.08)
PANEL_X = (-0.4, 20.0)  # x-range of every single panel, in swimmer sizes


def single_panel(y_lim):
    scale = PANEL["axes"] / (PANEL_X[1] - PANEL_X[0])
    axes_h = 2 * y_lim * scale
    height = PANEL["bottom"] + axes_h + PANEL["top"]
    fig = plt.figure(figsize=(PANEL["width"], height))
    ax = fig.add_axes([PANEL["left"] / PANEL["width"], PANEL["bottom"] / height,
                       PANEL["axes"] / PANEL["width"], axes_h / height])
    return fig, ax


def panel_legend(ax, handles=None, texts=None):
    if handles is None:
        handles, texts = ax.get_legend_handles_labels()
    ax.legend(handles, texts, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False,
              handlelength=1.0, borderaxespad=0)


def fig_axial(force, name="axial_pi3_paper_round", suffix=""):
    runs, p = paths(name)
    alpha = alpha_of(p)
    # Start angles as in the case file, -pi/4 ... pi/4.
    labels = [rf"$\beta_0 = {b}$" for b in
              (r"-\pi/4", r"-\pi/6", r"-\pi/12", "0", r"\pi/12", r"\pi/6", r"\pi/4")]
    sigmas = {k[0] for k in runs}
    for sigma, kind in ((max(sigmas), "pusher"), (min(sigmas), "puller")):
        fig, ax = single_panel(10)
        draw_paths(ax, runs, sigma, alpha, view=20, xlim=PANEL_X, ylim=(-10, 10), ends=False,
                   labels=labels)
        ax.set_yticks(np.arange(-10, 11, 5))
        # Legend from top to bottom in the order of the starts in the plot.
        handles, texts = ax.get_legend_handles_labels()
        panel_legend(ax, handles[::-1], texts[::-1])
        save(fig, f"fig_axial_{kind}_s{abs(sigma):g}{suffix}", force, exts=("png",), dpi=600)


def fig_transverse(force, name, suffix=""):
    runs, p = paths(name)
    alpha = alpha_of(p)
    # Headings as in the case file, -3pi/4 ... pi; the frame shows every wall contact.
    labels = [rf"$\vartheta = {t}$" for t in
              (r"-3\pi/4", r"-\pi/2", r"-\pi/4", "0", r"\pi/4", r"\pi/2", r"3\pi/4", r"\pi")]
    sigmas = {k[0] for k in runs}
    # Same frame for pusher and puller, at least +-14, large enough for every wall contact.
    y_wall = max(abs(r["y"][-1]) for r in runs.values() if r["end"] == "wall")
    y_lim = max(14, np.ceil(y_wall + 0.5))
    for sigma, kind in ((max(sigmas), "pusher"), (min(sigmas), "puller")):
        fig, ax = single_panel(y_lim)
        draw_paths(ax, runs, sigma, alpha, view=25, labels=labels,
                   xlim=PANEL_X, ylim=(-y_lim, y_lim), ends=False)
        # All paths share the start point: one black dot instead of overlapping coloured ones.
        r = runs[sigma, 0]
        ax.plot(r["x"][0], r["y"][0], "o", color="k", ms=3.5, zorder=4)
        panel_legend(ax)
        save(fig, f"fig_transverse_{kind}_s{abs(sigma):g}{suffix}", force, exts=("png",), dpi=600)


# Transverse case of the paper figures (sigma/(v0 l^2) = +-10).
TRANSVERSE_CASES = ["transverse_pi3_paper_theta_s10"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="redraw figures that already exist")
    args = ap.parse_args(argv)
    # Typeset with LaTeX (Computer Modern, as in REVTeX).
    with plt.rc_context({**STYLE, **TEX}):
        fig_axial(args.force)
        for name in TRANSVERSE_CASES:
            fig_transverse(args.force, name)


if __name__ == "__main__":
    main()
