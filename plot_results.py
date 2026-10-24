#!/usr/bin/env python3
"""
Plot base-model results and check them against Chavlis & Poirazi (2025).

Reads every results/*.json written by base_model.py, averages runs that differ
only by seed, and writes to figures/:

    1_paper_comparison.png    best test accuracy and loss per sampling type vs the
                              paper's Table 1 (mean +/- SD over seeds)
    2_size_sweep.png          test accuracy and loss vs trainable parameters
                              (the paper's Fig. 2b-c)
    3_dendrites_per_soma.png  test accuracy and loss vs dendrites per soma, one line
                              per soma count (the paper's Fig. 2d-e)
    4_learning_curves.png     train/validation curves for the best model of each
                              sampling type (checks convergence and overfitting)
    summary.csv               the numbers behind figure 1
    7_control_homogeneous.png control 1: V1 mixed dendrites vs each single nonlinearity
                              (LeakyReLU, supra-only, sub-only, Ca-only) vs model size
    control1_homogeneous.csv  at each size, V1 mixed vs the best single nonlinearity, with
                              a verdict ("mixed wins" / "single is enough")

Runs made with `base_model.py --nonlin ...` (tagged dann_<nonlin>_*) are grouped separately
from the LeakyReLU baseline. Figures 2-3 draw the baseline and the V1 mixed model (same
sampling color, dashed line and square markers for V1). Figure 1 compares the baseline with
the paper and adds a square "Yours (V1)" marker when V1 runs exist. Figure 4 has one row per
model. Figure 7 and the control CSV compare all nonlinearities, colored by nonlinearity.


    Pass this block of code in your terminals

    caffeinate -i bash -c '
    for seed in 1 2 3 4 5; do
    for cfg in "1 256" "4 256" "16 256" "64 256" "16 512"; do
        read d s <<< "$cfg"
        out="results/dann_local_fmnist_L1_d${d}_s${s}_seed${seed}.json"
        if [ -f "$out" ]; then echo "skip $out (already done)"; continue; fi
        echo "=== seed $seed: $d dendrites x $s somata ==="
        python base_model.py --dataset fmnist --dends $d --soma $s --seed $seed
    done
    done' 2>&1 | tee sweep.log
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paper reference values: Chavlis & Poirazi (2025), Table 1
# Mean +/- SD over N = 5 seeds, for the model size with the best test accuracy.
# Test loss is the loss of that same model. Accuracy in %.
# ---------------------------------------------------------------------------
PAPER = {
    "fmnist": {
        "random": {"acc": (89.612, 0.0870), "loss": (0.3245, 0.0028)},
        "local": {"acc": (90.256, 0.2237), "loss": (0.2975, 0.0042)},
        "global": {"acc": (90.182, 0.2164), "loss": (0.2947, 0.0052)},
    },
    "cifar10": {
        "random": {"acc": (52.458, 0.4347), "loss": (1.4612, 0.0172)},
        "local": {"acc": (56.966, 0.9796), "loss": (1.2684, 0.0230)},
        "global": {"acc": (56.998, 0.4875), "loss": (1.3398, 0.0217)},
    },
}
DATASET_NAME = {"fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
SAMPLING_NAME = {"local": "dANN-LRF", "global": "dANN-GRF", "random": "dANN-R"}
SAMPLING_ORDER = ["local", "global", "random"]
# Dendritic nonlinearity (base_model.py --nonlin). Result files that predate the flag are 'leaky'.
NONLIN_ORDER = ["leaky", "supra", "sub", "ca", "v1"]
NONLIN_NAME = {"leaky": "", "supra": " supra-only", "sub": " sub-only", "ca": " Ca-only",
               "v1": " + V1 mixed"}  # suffix on the sampling name
SINGLE_NONLINS = ["leaky", "supra", "sub", "ca"]  # the homogeneous-type controls (control 1)
SWEEP_NONLINS = ["leaky", "v1"]  # what the size-sweep figures (2, 3) draw
# In the sweep figures color follows the sampling type and the nonlinearity is encoded by
# line style and marker. In the control figure color follows the nonlinearity instead.
NONLIN_STYLE = {"leaky": dict(ls="-", marker="o"), "v1": dict(ls=(0, (2, 1.5)), marker="s"),
                "supra": dict(ls=":", marker="^"), "sub": dict(ls="-.", marker="v"),
                "ca": dict(ls=(0, (6, 2)), marker="D")}


def model_name(sampling, nonlin):
    return SAMPLING_NAME[sampling] + NONLIN_NAME[nonlin]

# Settings the paper used; runs with anything else are left out of the comparison.
PAPER_SETTINGS = {"layers": 1, "synapses": 16, "lr": 0.001, "batch_size": 128}


# ---------------------------------------------------------------------------
# Style: one validated palette, recessive chrome, ink-colored text
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots, fixed per sampling type (color follows the entity, never its rank)
COLOR = {"local": "#2a78d6", "global": "#eb6834", "random": "#1baf7a"}
# Control figure: one color per nonlinearity (same channel colors as analyze_v1.py)
NONLIN_COLOR = {"leaky": "#898781", "supra": "#eb6834", "sub": "#2a78d6", "ca": "#1baf7a",
                "v1": "#0b0b0b"}
TRAIN_VAL = {"train": "#2a78d6", "val": "#eb6834"}
# Ordinal blue ramp for soma counts, light -> dark (light end still clears 2:1)
SOMA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.edgecolor": AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_2,
})


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------

def load_runs(results_dir, dataset=None):
    runs = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            r = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skipping {path.name}: {e}")
            continue
        if "test_acc" not in r or "sampling" not in r:
            continue
        if dataset and r.get("dataset") != dataset:
            continue
        runs.append(r)
    return runs


def config_key(r):
    """Everything except the seed: runs with the same key are repeats of one model.

    Old result files predate --nonlin; they are the LeakyReLU baseline.
    """
    return (r["dataset"], r["sampling"], r["layers"], r["dends"], r["soma"],
            r["synapses"], r["lr"], r["batch_size"], r["epochs"], r.get("nonlin", "leaky"))


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def group_runs(runs):
    """Average over seeds. Returns a list of dicts, one per model configuration."""
    groups = defaultdict(list)
    for r in runs:
        groups[config_key(r)].append(r)

    configs = []
    for key, rs in groups.items():
        dataset, sampling, layers, dends, soma, synapses, lr, bs, epochs, nonlin = key
        acc_m, acc_s = mean_sd([100 * r["test_acc"] for r in rs])
        loss_m, loss_s = mean_sd([r["test_loss"] for r in rs])
        n_ep = min(len(r["train_loss"]) for r in rs)
        curves = {k: np.array([r[k][:n_ep] for r in rs], dtype=float)
                  for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
        configs.append(dict(
            dataset=dataset, sampling=sampling, layers=layers, dends=dends, soma=soma,
            synapses=synapses, lr=lr, batch_size=bs, epochs=epochs, nonlin=nonlin,
            n_params=rs[0]["n_params"], n=len(rs), seeds=sorted(r["seed"] for r in rs),
            acc=acc_m, acc_sd=acc_s, loss=loss_m, loss_sd=loss_s, curves=curves,
        ))
    return configs


def paper_comparable(c):
    return all(c[k] == v for k, v in PAPER_SETTINGS.items())


def best_configs(configs):
    """Best mean test accuracy per (dataset, sampling, nonlin) - the same rule as Table 1."""
    best = {}
    for c in configs:
        k = (c["dataset"], c["sampling"], c["nonlin"])
        if k not in best or c["acc"] > best[k]["acc"]:
            best[k] = c
    return best


def size_label(c):
    return f"{c['dends']} dends x {c['soma']} somata"


def verdict(c, paper):
    """Is the gap from the paper's mean within run-to-run noise?"""
    p_mean, p_sd = paper["acc"]
    diff = c["acc"] - p_mean
    # Two combined SDs, but never tighter than 0.5 percentage points
    tol = max(2 * math.sqrt(c["acc_sd"] ** 2 + p_sd ** 2), 0.5)
    if abs(diff) <= tol:
        return diff, tol, "matches paper"
    return diff, tol, "higher than paper" if diff > 0 else "lower than paper"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _end_labels(ax, items, min_gap=0.07):
    """Direct-label line ends, nudging labels apart so they never overlap.

    items: (x, y, text) in data coordinates. Works in axes-fraction space, so call
    it after everything is plotted; it freezes the y-limits first.
    """
    if not items:
        return
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi)
    log = ax.get_yscale() == "log"
    to_frac = (lambda y: (np.log10(y) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))) if log \
        else (lambda y: (y - lo) / (hi - lo))
    placed = sorted(([x, to_frac(y), t] for x, y, t in items), key=lambda p: p[1])
    original = [p[1] for p in placed]
    for i in range(1, len(placed)):  # push upward until clear of the label below
        placed[i][1] = max(placed[i][1], placed[i - 1][1] + min_gap)
    # re-center the stack on the lines it labels, then keep it inside the axes
    shift = np.mean(original) - np.mean([p[1] for p in placed])
    shift = min(max(shift, 0.02 - placed[0][1]), 0.98 - placed[-1][1])
    for p in placed:
        p[1] += shift
    for x, f, t in placed:
        ax.annotate(t, (x, f), xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2,
                    annotation_clip=False)


def _plain_log_ticks(axis):
    """Log axis with plain numbers (2,000 / 10,000) instead of 2x10^3, no minor labels."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _finish(fig, path, note=None):
    fig.tight_layout()
    if note:  # below the axes; bbox_inches="tight" grows the canvas to fit it
        fig.text(0.01, -0.01, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_paper_comparison(best, datasets, out):
    """Dot plot: your best model vs Table 1, for accuracy and loss."""
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.9 * len(datasets)),
                             squeeze=False)
    for row, ds in enumerate(datasets):
        # The paper comparison is baseline (leaky) vs Table 1; V1 runs are added as a third marker.
        samplings = [s for s in SAMPLING_ORDER
                     if (ds, s, "leaky") in best and s in PAPER.get(ds, {})]
        x = np.arange(len(samplings))
        for col, (metric, label) in enumerate([("acc", "Test accuracy (%)"),
                                               ("loss", "Test loss (cross-entropy)")]):
            ax = axes[row, col]
            v1_labelled = False
            for i, s in enumerate(samplings):
                c = best[(ds, s, "leaky")]
                p_mean, p_sd = PAPER[ds][s][metric]
                y_mean, y_sd = c[metric], c[f"{metric}_sd"]
                # Paper: hollow gray diamond. Yours: filled circle in the sampling color.
                ax.errorbar(i - 0.12, p_mean, yerr=p_sd, fmt="D", ms=7, mfc=SURFACE,
                            mec=INK_2, mew=1.6, ecolor=INK_2, elinewidth=1.4, capsize=3,
                            label="Paper (Table 1)" if i == 0 else None, zorder=3)
                ax.errorbar(i + 0.12, y_mean, yerr=y_sd, fmt="o", ms=7, mfc=COLOR[s],
                            mec=SURFACE, mew=1.5, ecolor=COLOR[s], elinewidth=1.4,
                            capsize=3, label="Yours" if i == 0 else None, zorder=4)
                fmt = "{:.2f}" if metric == "acc" else "{:.3f}"
                ax.annotate(fmt.format(y_mean), (i + 0.12, y_mean), xytext=(9, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
                ax.annotate(fmt.format(p_mean), (i - 0.12, p_mean), xytext=(-9, 0),
                            textcoords="offset points", va="center", ha="right",
                            fontsize=9, color=INK_2)
                if (ds, s, "v1") in best:  # V1 dendrites: filled square, same color
                    v = best[(ds, s, "v1")]
                    ax.errorbar(i + 0.36, v[metric], yerr=v[f"{metric}_sd"], fmt="s", ms=7,
                                mfc=COLOR[s], mec=SURFACE, mew=1.5, ecolor=COLOR[s],
                                elinewidth=1.4, capsize=3,
                                label=None if v1_labelled else "Yours (V1 dendrites)", zorder=4)
                    v1_labelled = True
                    ax.annotate(fmt.format(v[metric]), (i + 0.36, v[metric]), xytext=(0, -13),
                                textcoords="offset points", va="top", ha="center",
                                fontsize=9, color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{SAMPLING_NAME[s]}\n{best[(ds, s, 'leaky')]['dends']} x "
                                f"{best[(ds, s, 'leaky')]['soma']}\n"
                                f"{best[(ds, s, 'leaky')]['n']} seed(s)"
                                for s in samplings], fontsize=9)
            ax.set_xlabel("Best model (dendrites per soma x somata)")
            ax.set_xlim(-0.6, len(samplings) - 0.4)
            ax.set_ylabel(label)
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.25)
            title = "accuracy" if metric == "acc" else "loss (lower is better)"
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}: best model, test {title}")
            if row == 0 and col == 0:
                ax.legend(loc="lower left")
    _finish(fig, out / "1_paper_comparison.png",
            "Error bars: SD over seeds (n = seeds you ran; paper n = 5). Your best size is "
            "picked the paper's way: highest mean test accuracy.")


def fig_size_sweep(configs, datasets, out):
    """Test accuracy and loss vs trainable parameters (log x), one line per sampling."""
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.8 * len(datasets)),
                             squeeze=False)
    for row, ds in enumerate(datasets):
        for col, (metric, label) in enumerate([("acc", "Test accuracy (%)"),
                                               ("loss", "Test loss")]):
            ax = axes[row, col]
            ends = []
            for s in SAMPLING_ORDER:
                any_line = False
                for n in SWEEP_NONLINS:
                    cs = sorted((c for c in configs if c["dataset"] == ds
                                 and c["sampling"] == s and c["nonlin"] == n),
                                key=lambda c: c["n_params"])
                    if not cs:
                        continue
                    any_line = True
                    px = np.array([c["n_params"] for c in cs])
                    m = np.array([c[metric] for c in cs])
                    sd = np.array([c[f"{metric}_sd"] for c in cs])
                    ax.fill_between(px, m - sd, m + sd, color=COLOR[s], alpha=0.15, lw=0)
                    ax.plot(px, m, color=COLOR[s], mec=SURFACE, mew=1.2,
                            label=model_name(s, n), **NONLIN_STYLE[n])
                    ends.append((px[-1], m[-1], model_name(s, n)))
                if any_line and ds in PAPER and s in PAPER[ds]:
                    ax.axhline(PAPER[ds][s][metric][0], color=COLOR[s], lw=1, ls=(0, (4, 3)))
            ax.set_xscale("log")
            _plain_log_ticks(ax.xaxis)
            ax.margins(x=0.08)
            _end_labels(ax, ends)
            ax.set_xlabel("Trainable parameters (log scale)")
            ax.set_ylabel(label)
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}: test "
                         f"{'accuracy' if metric == 'acc' else 'loss'} vs model size")
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
                labels.append("Paper best (Table 1)")
                ax.legend(handles, labels, loc="lower right")
    _finish(fig, out / "2_size_sweep.png",
            "Compare with the paper's Fig. 2b-c. Shading: SD over seeds. Dashed: the paper's "
            "best value for that sampling type, in the same color.")


def fig_dendrites_per_soma(configs, datasets, out):
    """Paper Fig. 2d-e: x = dendrites per soma, one line per soma count."""
    panels = []
    for ds in datasets:
        for s in SAMPLING_ORDER:
            for n in SWEEP_NONLINS:
                cs = [c for c in configs if c["dataset"] == ds and c["sampling"] == s
                      and c["nonlin"] == n]
                by_soma = defaultdict(list)
                for c in cs:
                    by_soma[c["soma"]].append(c)
                if any(len(v) >= 2 for v in by_soma.values()):
                    panels.append((ds, s, n, by_soma))
    if not panels:
        print("  skipping 3_dendrites_per_soma.png: need >= 2 dendrite counts at one soma count")
        return

    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.6 * len(panels)), squeeze=False)
    for row, (ds, s, n, by_soma) in enumerate(panels):
        somas = sorted(by_soma)
        # spread the ordinal ramp across however many soma counts there are
        steps = np.linspace(0, len(SOMA_RAMP) - 1, len(somas)).round().astype(int)
        for col, (metric, label) in enumerate([("acc", "Test accuracy (%)"),
                                               ("loss", "Test loss")]):
            ax = axes[row, col]
            ends = []
            for soma, step in zip(somas, steps):
                cs = sorted(by_soma[soma], key=lambda c: c["dends"])
                if len(cs) < 2:
                    continue
                dx = np.array([c["dends"] for c in cs])
                m = np.array([c[metric] for c in cs])
                sd = np.array([c[f"{metric}_sd"] for c in cs])
                color = SOMA_RAMP[step]
                ax.fill_between(dx, m - sd, m + sd, color=color, alpha=0.15, lw=0)
                ax.plot(dx, m, "-o", color=color, mec=SURFACE, mew=1.2, label=f"{soma} somata")
                ends.append((dx[-1], m[-1], f"{soma} somata"))
            if ds in PAPER and s in PAPER[ds]:
                ax.axhline(PAPER[ds][s][metric][0], color=MUTED, lw=1, ls=(0, (4, 3)),
                           label="Paper best (Table 1)")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.margins(x=0.08)
            _end_labels(ax, ends)
            ax.set_xlabel("Dendrites per soma (log2 scale)")
            ax.set_ylabel(label)
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {model_name(s, n)}: test "
                         f"{'accuracy' if metric == 'acc' else 'loss'}")
            if col == 0:
                ax.legend(loc="lower right")
    _finish(fig, out / "3_dendrites_per_soma.png",
            "Compare with the paper's Fig. 2d-e (Fashion-MNIST, 256 and 512 somata): accuracy "
            "should rise and loss fall as dendrites per soma increase.")


def fig_learning_curves(best, out):
    """Train vs validation curves for the best model per (dataset, sampling)."""
    keys = sorted(best, key=lambda k: (k[0], SAMPLING_ORDER.index(k[1]),
                                       NONLIN_ORDER.index(k[2])))
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 3.2 * len(keys)), squeeze=False)
    for row, k in enumerate(keys):
        c = best[k]
        ep = np.arange(1, c["curves"]["train_loss"].shape[1] + 1)
        for col, (metric, label, scale) in enumerate([("acc", "Accuracy (%)", 100),
                                                      ("loss", "Loss", 1)]):
            ax = axes[row, col]
            ends = []
            for split in ("train", "val"):
                y = c["curves"][f"{split}_{metric}"] * scale
                m, sd = y.mean(0), (y.std(0, ddof=1) if len(y) > 1 else np.zeros(y.shape[1]))
                ax.fill_between(ep, m - sd, m + sd, color=TRAIN_VAL[split], alpha=0.15, lw=0)
                name = "Train" if split == "train" else "Validation"
                ax.plot(ep, m, color=TRAIN_VAL[split], label=name)
                ends.append((ep[-1], m[-1], name))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            test = c[metric]
            ax.axhline(test, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.annotate(f"test {test:.2f}" if metric == "acc" else f"test {test:.3f}",
                        (ep[0], test), xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=INK_2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_xlim(1, ep[-1] + max(1, 0.12 * ep[-1]))
            _end_labels(ax, ends)
            ds_name = DATASET_NAME.get(k[0], k[0])
            if col == 0:
                ax.set_title(f"{ds_name}, {model_name(k[1], k[2])}: accuracy", pad=18)
                ax.text(0, 1.02, f"best model: {size_label(c)}, n={c['n']} seed(s)",
                        transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
            else:
                ax.set_title("Loss", pad=18)
            if row == 0 and col == 0:
                ax.legend(loc="lower right")
    _finish(fig, out / "4_learning_curves.png",
            "A validation curve that rises while training keeps falling (loss) signals "
            "overfitting. The paper reports dANNs overfit far less than vanilla ANNs.")


# ---------------------------------------------------------------------------
# Control 1: homogeneous type ("any single nonlinearity is enough")
# ---------------------------------------------------------------------------

NONLIN_LABEL = {"leaky": "LeakyReLU (paper)", "supra": "supra-only", "sub": "sub-only",
                "ca": "Ca-only", "v1": "V1 mixed"}


def fig_control_homogeneous(configs, datasets, out):
    """Mixed V1 dendrites vs each single nonlinearity, test metrics vs model size."""
    panels = [(ds, s) for ds in datasets for s in SAMPLING_ORDER
              if any(c["dataset"] == ds and c["sampling"] == s and c["nonlin"] == "v1"
                     for c in configs)]
    if not panels:
        print("  skipping 7_control_homogeneous.png: no --nonlin v1 runs")
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.8 * len(panels)), squeeze=False)
    for row, (ds, s) in enumerate(panels):
        for col, (metric, label) in enumerate([("acc", "Test accuracy (%)"),
                                               ("loss", "Test loss")]):
            ax = axes[row, col]
            ends = []
            for n in NONLIN_ORDER:
                cs = sorted((c for c in configs if c["dataset"] == ds and c["sampling"] == s
                             and c["nonlin"] == n), key=lambda c: c["n_params"])
                if not cs:
                    continue
                px = np.array([c["n_params"] for c in cs])
                m = np.array([c[metric] for c in cs])
                sd = np.array([c[f"{metric}_sd"] for c in cs])
                ax.fill_between(px, m - sd, m + sd, color=NONLIN_COLOR[n], alpha=0.12, lw=0)
                # error bars as well: with one size per model there is no band to shade
                ax.errorbar(px, m, yerr=sd, fmt="none", ecolor=NONLIN_COLOR[n], elinewidth=1.2,
                            capsize=3, alpha=0.8, zorder=2)
                ax.plot(px, m, color=NONLIN_COLOR[n], marker=NONLIN_STYLE[n]["marker"],
                        mec=SURFACE, mew=1.2, lw=2.6 if n == "v1" else 2, label=NONLIN_LABEL[n],
                        zorder=3)
                ends.append((px[-1], m[-1], NONLIN_LABEL[n]))
            if ds in PAPER and s in PAPER[ds]:
                ax.axhline(PAPER[ds][s][metric][0], color=MUTED, lw=1, ls=(0, (4, 3)),
                           label="Paper best (Table 1)")
            ax.set_xscale("log")
            _plain_log_ticks(ax.xaxis)
            ax.margins(x=0.08)
            _end_labels(ax, ends)
            ax.set_xlabel("Trainable parameters (log scale)")
            ax.set_ylabel(label)
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {SAMPLING_NAME[s]}: control 1, test "
                         f"{'accuracy' if metric == 'acc' else 'loss'}")
            if row == 0 and col == 0:
                ax.legend(loc="lower right")
    _finish(fig, out / "7_control_homogeneous.png",
            "Control 1 (homogeneous type): if any single-nonlinearity line matches the black "
            "V1 mixed line, mixing adds nothing. Single-nonlinearity models have the baseline's "
            "parameter count; V1 mixed has three cable weights per dendrite. Error bars and "
            "shading: SD over seeds.")


def write_control_homogeneous(configs, out):
    """control1_homogeneous.csv: at each exact size, V1 mixed vs the best single nonlinearity."""
    by_size = defaultdict(dict)
    for c in configs:
        by_size[(c["dataset"], c["sampling"], c["dends"], c["soma"])][c["nonlin"]] = c
    rows = []
    for key in sorted(by_size, key=lambda k: (k[0], SAMPLING_ORDER.index(k[1]), k[3], k[2])):
        group = by_size[key]
        singles = [n for n in SINGLE_NONLINS if n in group]
        if "v1" not in group or not singles:
            continue
        ds, s, dends, soma = key
        v = group["v1"]
        best_n = max(singles, key=lambda n: group[n]["acc"])
        b = group[best_n]
        gap = v["acc"] - b["acc"]
        # Same rule as verdict(): two combined SDs, never tighter than 0.5 percentage points
        tol = max(2 * math.sqrt(v["acc_sd"] ** 2 + b["acc_sd"] ** 2), 0.5)
        row = {"dataset": ds, "model": SAMPLING_NAME[s], "size": f"{dends} dends x {soma} somata",
               "seeds": min(c["n"] for c in group.values())}
        for n in NONLIN_ORDER:
            row[f"{n}_acc"] = round(group[n]["acc"], 3) if n in group else ""
            row[f"{n}_acc_sd"] = round(group[n]["acc_sd"], 3) if n in group else ""
        row.update(best_single=best_n, gap_pp=round(gap, 3), tolerance_pp=round(tol, 3),
                   verdict="mixed wins" if gap > tol else "single is enough")
        rows.append(row)
    if not rows:
        print("  skipping control1_homogeneous.csv: need v1 and single-nonlinearity runs "
              "at the same size")
        return rows
    path = out / "control1_homogeneous.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")

    print("\nControl 1 (homogeneous type): V1 mixed vs single nonlinearities, test accuracy (%)")
    cols = "".join(f"{NONLIN_LABEL[n]:>18s}" for n in NONLIN_ORDER)
    print(f"{'dataset':9s} {'model':9s} {'size':22s} {'seeds':>5s}{cols}  {'best single':>11s} "
          f"{'gap':>6s}  verdict")
    for r in rows:
        cells = "".join(f"{r[f'{n}_acc']:7.2f} +/- {r[f'{n}_acc_sd']:5.2f}" if r[f"{n}_acc"] != ""
                        else f"{'-':>18s}" for n in NONLIN_ORDER)
        print(f"{r['dataset']:9s} {r['model']:9s} {r['size']:22s} {r['seeds']:>5d}{cells}  "
              f"{r['best_single']:>11s} {r['gap_pp']:+6.2f}  {r['verdict']}")
    if any(r["seeds"] < 3 for r in rows):
        print("  note: fewer than 3 seeds, so the SD and the verdict are unreliable; "
              "the tolerance floor of 0.5 pp is doing the work.")
    return rows


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary(best, n_configs, out):
    rows = []
    order = lambda kv: (kv[0][0], SAMPLING_ORDER.index(kv[0][1]), NONLIN_ORDER.index(kv[0][2]))
    for (ds, s, n), c in sorted(best.items(), key=order):
        if ds not in PAPER or s not in PAPER[ds]:
            continue
        p = PAPER[ds][s]
        diff, tol, v = verdict(c, p)
        rows.append({
            "dataset": ds, "model": model_name(s, n), "nonlin": n, "best_size": size_label(c),
            "n_params": c["n_params"], "sizes_tried": n_configs[(ds, s, n)],
            "seeds": c["n"], "epochs": c["epochs"],
            "your_acc": round(c["acc"], 3), "your_acc_sd": round(c["acc_sd"], 3),
            "paper_acc": p["acc"][0], "paper_acc_sd": p["acc"][1],
            "diff_pp": round(diff, 3), "tolerance_pp": round(tol, 3),
            "your_loss": round(c["loss"], 4), "paper_loss": p["loss"][0],
            "verdict": v,
        })
    if not rows:
        return rows
    path = out / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")
    return rows


def print_summary(rows):
    print("\nBest model vs paper Table 1 (test accuracy, %)")
    print(f"{'dataset':9s} {'model':20s} {'best size':22s} {'seeds':>5s}  "
          f"{'yours':>14s}  {'paper':>14s}  {'diff':>6s}  verdict")
    for r in rows:
        yours = f"{r['your_acc']:.2f} +/- {r['your_acc_sd']:.2f}"
        paper = f"{r['paper_acc']:.2f} +/- {r['paper_acc_sd']:.2f}"
        print(f"{r['dataset']:9s} {r['model']:20s} {r['best_size']:22s} {r['seeds']:>5d}  "
              f"{yours:>14s}  {paper:>14s}  {r['diff_pp']:+6.2f}  {r['verdict']}")
    notes = []
    if any(r["seeds"] < 5 for r in rows):
        notes.append("Fewer than 5 seeds: your SD is less reliable than the paper's "
                     "(and 0 with one seed).")
    paper_epochs = {"fmnist": 25, "cifar10": 50}
    if any(r["epochs"] < paper_epochs.get(r["dataset"], 0) for r in rows):
        notes.append("Some best models trained for fewer epochs than the paper's code "
                     "(25 Fashion-MNIST, 50 CIFAR-10), so expect lower accuracy.")
    if any(r["sizes_tried"] < 5 for r in rows):
        notes.append("Few model sizes tried: the paper's best came from a sweep of 1-64 "
                     "dendrites x 32-512 somata, so a narrow sweep can land slightly below it.")
    if any(r["model"].startswith("dANN-GRF") for r in rows):
        notes.append("dANN-GRF: the paper's code left dendrites beyond 9 per soma without "
                     "inputs; base_model.py does not, so GRF can differ from the paper.")
    for n in notes:
        print(f"  note: {n}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot base_model.py results against the paper")
    p.add_argument("--results", default="results", help="folder with the .json results")
    p.add_argument("--out", default="figures", help="folder to write figures into")
    p.add_argument("--dataset", default=None, choices=["fmnist", "cifar10"])
    args = p.parse_args()

    runs = load_runs(args.results, args.dataset)
    if not runs:
        raise SystemExit(f"No results found in {args.results}/. Run base_model.py first.")
    configs = group_runs(runs)
    comparable = [c for c in configs if paper_comparable(c)]
    skipped = len(configs) - len(comparable)
    print(f"Loaded {len(runs)} runs = {len(configs)} model configurations"
          + (f" ({skipped} left out: settings differ from the paper's)" if skipped else ""))
    if not comparable:
        raise SystemExit("No runs use the paper's settings "
                         f"({PAPER_SETTINGS}); nothing to compare.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in ("fmnist", "cifar10") if any(c["dataset"] == d for c in comparable)]
    best = best_configs(comparable)
    n_configs = defaultdict(int)
    for c in comparable:
        n_configs[(c["dataset"], c["sampling"], c["nonlin"])] += 1

    fig_paper_comparison(best, datasets, out)
    fig_size_sweep(comparable, datasets, out)
    fig_dendrites_per_soma(comparable, datasets, out)
    fig_learning_curves(best, out)
    fig_control_homogeneous(comparable, datasets, out)
    print_summary(write_summary(best, n_configs, out))
    write_control_homogeneous(comparable, out)


if __name__ == "__main__":
    main()
