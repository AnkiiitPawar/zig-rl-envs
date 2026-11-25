"""Figures for the README.

    python3 scripts/bench_envs.py   # docs/data/bench_envs.json
    python3 scripts/bench_ppo.py    # docs/data/bench_ppo.json and .cache/ppo/*.csv
    python3 scripts/figures.py
"""
import csv
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from common import CACHE, FIGURES, RESULTS  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, ORANGE_LIGHT, AQUA_LIGHT = "#d6e6f8", "#fbe0d4", "#d2f0e4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
TITLES = {"cartpole": "CartPole-v1", "acrobot": "Acrobot-v1", "pendulum": "Pendulum-v1"}
TARGETS = {"cartpole": 475, "acrobot": -100, "pendulum": -200}


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def curves():
    """Learning curves of every seed, with the median."""
    bench = json.loads((RESULTS / "bench_ppo.json").read_text())
    seeds = bench["seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, env in zip(axes, ["cartpole", "acrobot", "pendulum"]):
        grid = None
        ys = []
        for s in range(1, seeds + 1):
            with open(CACHE / "ppo" / f"{env}-{s}.csv") as fh:
                rows = [r for r in csv.DictReader(fh) if r["mean_return_100"] != "nan"]
            x = np.array([float(r["step"]) for r in rows])
            y = np.array([float(r["mean_return_100"]) for r in rows])
            ax.plot(x, y, color=BLUE, alpha=0.25, lw=0.9)
            grid = np.linspace(x[0], x[-1], 300) if grid is None else grid
            ys.append(np.interp(grid, x, y))
        ax.plot(grid, np.median(ys, axis=0), color=BLUE, lw=2.2)
        ax.axhline(TARGETS[env], color=ORANGE, lw=1.2, ls=(0, (4, 3)))
        runs = [r for r in bench["runs"] if r["env"] == env]
        solved = [r["solved_seconds"] for r in runs if r["solved_seconds"] is not None]
        med = np.median(solved) if solved else float("nan")
        ax.text(0.98, 0.04, f"target {TARGETS[env]}: reached by {len(solved)} of {len(runs)} seeds,\n"
                f"median {med:.1f} s on one thread", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=INK2)
        ax.set_title(TITLES[env])
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    axes[0].set_ylabel("mean return, last 100 episodes")
    fig.suptitle(f"PPO in zrl: every seed (thin) and the median (thick), {seeds} seeds per environment",
                 x=0.06, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "curves.png")


def speed():
    """Environment steps per second against the number of environments."""
    bench = json.loads((RESULTS / "bench_envs.json").read_text())
    styles = {"zrl": (BLUE, "-", "zrl (Zig)"), "envpool": (AQUA, "-", "EnvPool (C++)"),
              "gymnasium": (ORANGE, "-", "Gymnasium"), "gym-numpy": (ORANGE, (0, (4, 2)), "Gymnasium, NumPy-vectorised")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
    for ax, env in zip(axes, ["cartpole", "acrobot", "pendulum"]):
        for impl, (col, ls, label) in styles.items():
            pts = sorted((r["envs"], r["steps_per_second"]) for r in bench["runs"] if r["env"] == env and r["impl"] == impl)
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=col, ls=ls, lw=2, marker="o", ms=4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLES[env])
        ax.set_xlabel("environments stepped together")
    axes[0].set_ylabel("environment steps per second")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle("Environment steps per second with random actions and resets, one thread", x=0.06, ha="left",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.04)
    fig.text(0.06, 0.965, "zrl is timed inside Zig; the others are driven from Python, as they are used. EnvPool runs "
             "with one worker thread.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "speed.png")


def pipeline():
    """Diagram: how one batched step is computed."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.05, 4.85, "One step of N CartPoles: state in columns, one array per variable", fontsize=11,
            fontweight="bold", color=INK, va="top")
    names = ["x", "ẋ", "θ", "ω"]
    for j, nm in enumerate(names):  # the state columns
        x0 = 0.3 + j * 0.55
        for k in range(10):
            ax.add_patch(Rectangle((x0, 0.9 + k * 0.28), 0.42, 0.24, fc=BLUE_LIGHT, ec="none"))
        ax.text(x0 + 0.21, 0.7, nm, ha="center", va="top", fontsize=10, color=INK)
    ax.text(1.35, 3.95, "state, float64\n(N envs; ω = dθ/dt)", ha="center", fontsize=8.5, color=INK2)

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 0.15, y + h - 0.2, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.15, y + h - 0.55, body, fontsize=8.5, color=INK2, va="top", linespacing=1.45)

    box(2.75, 2.55, 3.4, 1.75, "1. Library pass", "for each env: cos θ, sin θ\nfrom glibc's libm, one value at\na time, the same bits NumPy gets",
        ORANGE_LIGHT, ORANGE)
    box(2.75, 0.55, 3.4, 1.75, "2. Arithmetic pass", "the accelerations and the Euler\nupdate on 8 envs per @Vector(8, f64),\neach operation in Gymnasium's order",
        BLUE_LIGHT, BLUE)
    box(6.75, 1.55, 2.6, 1.75, "3. Rewards and ends", "reward, terminated (|x| > 2.4\nor |θ| > 12°), truncated\nat 500 steps", AQUA_LIGHT, AQUA)
    box(9.8, 1.55, 2.4, 1.75, "4. Reset", "finished envs: keep the\nlast observation, draw a\nnew start", AQUA_LIGHT, AQUA)
    for a, b in [((2.55, 2.9), (2.75, 3.3)), ((2.55, 1.9), (2.75, 1.5)), ((4.45, 2.55), (4.45, 2.3)), ((6.15, 1.45), (6.75, 2.2)),
                 ((9.35, 2.4), (9.8, 2.4))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
    ax.text(6.75, 0.75, "Observations go out as float32, as Gymnasium returns them.", fontsize=8.5, color=MUTED)
    save(fig, "pipeline.png")


def main():
    which = sys.argv[1:] or ["pipeline", "curves", "speed"]
    if "pipeline" in which:
        pipeline()
    if "curves" in which:
        curves()
    if "speed" in which:
        speed()


if __name__ == "__main__":
    main()
