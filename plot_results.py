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
    8_ratio_sweep.png         per-dendrite ratio models (--nonlin branch --mix S:B:C) vs the
                              best single nonlinearity and V1 mixed at the same size
    ratio_sweep.csv           the numbers behind figure 8, with a verdict per ratio
    9_constants_sweep.png     for models whose fixed constants were varied (--leak,
                              --ca-thresh, ...): validation and test accuracy per variant
    constants_sweep.csv       the numbers behind figure 9, ranked by validation accuracy,
                              plus ready-to-paste sweep_constants.sh lines for the winners

Runs that change a fixed constant are separate variants named <model>@<settings>, e.g.
ca@thr0.5_leak0.1. --select chooses which variant of each model figures 7-8 and the control
tables show: the default constants (default) or the best VALIDATION accuracy (best-val).

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
NONLIN_ORDER = ["leaky", "supra", "sub", "ca", "v1", "gate"]
NONLIN_NAME = {"leaky": "", "supra": " supra-only", "sub": " sub-only", "ca": " Ca-only",
               "v1": " + V1 mixed", "gate": " + Ca gate"}  # suffix on the sampling name
SINGLE_NONLINS = ["leaky", "supra", "sub", "ca"]  # the homogeneous-type controls (control 1)
SWEEP_NONLINS = ["leaky", "v1"]  # what the size-sweep figures (2, 3) draw
# In the sweep figures color follows the sampling type and the nonlinearity is encoded by
# line style and marker. In the control figure color follows the nonlinearity instead.
NONLIN_STYLE = {"leaky": dict(ls="-", marker="o"), "v1": dict(ls=(0, (2, 1.5)), marker="s"),
                "supra": dict(ls=":", marker="^"), "sub": dict(ls="-.", marker="v"),
                "ca": dict(ls=(0, (6, 2)), marker="D"), "gate": dict(ls=(0, (1, 1)), marker="X")}

# --nonlin branch runs are one model variant per ratio, keyed "branch:<supra:sub:ca>".
BRANCH_PREFIX = "branch:"
BRANCH_STYLE = dict(ls=(0, (4, 1, 1, 1)), marker="P")
BRANCH_RAMP = ["#6a3fd6", "#9b7ae6", "#43278f", "#c3b1f0", "#2d1a66", "#d9ccf7"]
_BRANCH_VARIANTS = set()  # registered by group_runs so colors are stable across figures

# Fixed constants of the biological nonlinearities (mirror of base_model.py: keep in sync).
# A run with non-default constants is its own variant: "<base>@<suffix>", e.g. "ca@thr0.5_leak0.1"
# or "gate@bsupra_leak0.1" (basal nonlinearity first). The base name is the part before "@".
CONSTANT_DEFAULTS = dict(supra_p=2.0, sub_sat=1.0, ca_thresh=1.0, ca_gain=10.0, ca_amp=1.0,
                         leak=0.0, apical_frac=0.25)
CONSTANT_SHORT = dict(supra_p="p", sub_sat="sat", ca_thresh="thr", ca_gain="gain", ca_amp="amp",
                      leak="leak", apical_frac="ap")


def run_constants(r):
    """The constants a run used, with defaults filled in for files that predate a flag."""
    c = {k: (r.get(k) if r.get(k) is not None else d) for k, d in CONSTANT_DEFAULTS.items()}
    c["basal_nonlin"] = r.get("basal_nonlin") or "leaky"
    c["weight_decay"] = float(r.get("weight_decay") or 0.0)  # files before --weight-decay: 0
    c["apical_input"] = r.get("apical_input") or "same"      # files before --apical-input: same
    c["apical_pool"] = int(r.get("apical_pool") or 4)
    return c


def constants_parts(consts, nonlin):
    """['bsupra', 'thr0.5', 'leak0.1']: the non-default settings, in a fixed order."""
    parts = []
    if nonlin == "gate" and consts.get("basal_nonlin", "leaky") != "leaky":
        parts.append(f"b{consts['basal_nonlin']}")
    for key, default in CONSTANT_DEFAULTS.items():
        if nonlin == "leaky" or (key == "apical_frac" and nonlin != "gate"):
            continue  # the paper baseline has no nonlinearity constants
        if float(consts.get(key, default)) != default:
            parts.append(f"{CONSTANT_SHORT[key]}{float(consts[key]):g}")
    if consts.get("apical_input", "same") != "same":  # any model, the baseline included
        pool = consts.get("apical_pool", 4)
        parts.append(f"ap{consts['apical_input']}" + (f"{pool}" if pool != 4 else ""))
    if consts.get("weight_decay"):  # applies to every model, the baseline included
        parts.append(f"wd{float(consts['weight_decay']):g}")
    return parts


def base_of(n):
    return n.split("@", 1)[0]


def variant_suffix(n):
    return n.split("@", 1)[1] if "@" in n else ""


def is_branch(n):
    return base_of(n).startswith(BRANCH_PREFIX)


def branch_mix(n):
    return base_of(n)[len(BRANCH_PREFIX):]


def variant_of(r):
    """Model variant key for a run: nonlin name (or 'branch:<mix>'), plus '@<non-default constants>'."""
    n = r.get("nonlin", "leaky")
    base = f"{BRANCH_PREFIX}{r['mix']}" if n == "branch" else n
    parts = constants_parts(run_constants(r), n)
    return base + ("@" + "_".join(parts) if parts else "")


def mix_sort_key(mix):
    """Order ratios by their Ca share (12:3:1 before 1:1:1 before 1:3:12)."""
    parts = [float(p) for p in mix.split(":")]
    return (parts[2] / sum(parts), parts[1] / sum(parts), mix)


def nonlin_rank(n):
    b = base_of(n)
    if b in NONLIN_ORDER:
        return (NONLIN_ORDER.index(b), (0, 0, ""), variant_suffix(n))
    return (len(NONLIN_ORDER), mix_sort_key(branch_mix(b)), variant_suffix(n))


def _with_suffix(text, n):
    s = variant_suffix(n)
    return f"{text} [{s}]" if s else text


def nonlin_suffix(n):
    b = base_of(n)
    return _with_suffix(NONLIN_NAME[b] if b in NONLIN_NAME else f" branch {branch_mix(b)}", n)


def nonlin_label(n):
    b = base_of(n)
    return _with_suffix(NONLIN_LABEL[b] if b in NONLIN_LABEL else f"branch {branch_mix(b)}", n)


def nonlin_style(n):
    return NONLIN_STYLE.get(base_of(n), BRANCH_STYLE)


def nonlin_color(n):
    b = base_of(n)
    if b in NONLIN_COLOR:
        return NONLIN_COLOR[b]
    ranked = sorted(_BRANCH_VARIANTS | {b}, key=nonlin_rank)
    return BRANCH_RAMP[ranked.index(b) % len(BRANCH_RAMP)]


def model_name(sampling, nonlin):
    return SAMPLING_NAME[sampling] + nonlin_suffix(nonlin)

# Settings the paper used; runs with anything else are left out of the comparison.
PAPER_SETTINGS = {"layers": 1, "synapses": 16, "lr": 0.001, "batch_size": 128,
                  "train_subset": None}  # --train-subset runs are quick checks, not results


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
                "v1": "#0b0b0b", "gate": "#0f8b8d"}
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
            r["synapses"], r["lr"], r["batch_size"], r["epochs"], variant_of(r),
            r.get("train_subset"))  # variant_of already encodes every non-default constant


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
        dataset, sampling, layers, dends, soma, synapses, lr, bs, epochs, nonlin, subset = key
        if is_branch(nonlin):
            _BRANCH_VARIANTS.add(base_of(nonlin))
        acc_m, acc_s = mean_sd([100 * r["test_acc"] for r in rs])
        loss_m, loss_s = mean_sd([r["test_loss"] for r in rs])
        # final-epoch validation accuracy: the only number used to choose between variants
        val_m, val_s = mean_sd([100 * r["val_acc"][-1] for r in rs])
        n_ep = min(len(r["train_loss"]) for r in rs)
        curves = {k: np.array([r[k][:n_ep] for r in rs], dtype=float)
                  for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
        configs.append(dict(
            dataset=dataset, sampling=sampling, layers=layers, dends=dends, soma=soma,
            synapses=synapses, lr=lr, batch_size=bs, epochs=epochs, nonlin=nonlin,
            variant=nonlin, train_subset=subset, constants=run_constants(rs[0]),
            n_params=rs[0]["n_params"], n=len(rs), seeds=sorted(r["seed"] for r in rs),
            acc=acc_m, acc_sd=acc_s, loss=loss_m, loss_sd=loss_s, curves=curves,
            val_acc=val_m, val_acc_sd=val_s,
            val_by_seed={r["seed"]: 100 * r["val_acc"][-1] for r in rs},
            acc_by_seed={r["seed"]: 100 * r["test_acc"] for r in rs},
            # early-stopped test accuracy; only runs made after it was recorded have it
            acc_best_by_seed={r["seed"]: 100 * r["test_acc_best_val"] for r in rs
                              if "test_acc_best_val" in r},
        ))
    return configs


def selection_scores(cs):
    """Validation score used to choose between variants of one model, one per config in `cs`.

    Variants are compared on the seeds they all share (stage 1 trains every variant with the
    same seed), so a winner that later gets more seeds is not penalised or favoured against
    single-seed variants just because its mean now covers different seeds. With no shared
    seed the plain mean over each variant's own seeds is used.
    """
    common = set.intersection(*(set(c["val_by_seed"]) for c in cs)) if cs else set()
    if not common:
        return [c["val_acc"] for c in cs]
    return [float(np.mean([c["val_by_seed"][k] for k in common])) for c in cs]


def select_variants(configs, mode="default"):
    """One config per (dataset, sampling, size, base model) for the comparison figures.

    'default'  keeps the variant with default constants (tuned variants are left out).
    'best-val' keeps the variant with the highest validation accuracy on the seeds all
               variants share (see selection_scores). Test accuracy is never consulted, so
               the test numbers of the chosen variants stay honest.
    Returned configs have `nonlin` set to the base name and `variant` to the full key.
    """
    groups = defaultdict(list)
    for c in configs:
        groups[(c["dataset"], c["sampling"], c["dends"], c["soma"], base_of(c["nonlin"]))].append(c)
    chosen = []
    for (ds, s, d, so, base), cs in groups.items():
        if mode == "best-val":
            pick = cs[int(np.argmax(selection_scores(cs)))]
        else:
            plain = [c for c in cs if c["nonlin"] == base]
            if not plain:
                continue
            pick = plain[0]
        chosen.append(dict(pick, nonlin=base, variant=pick["nonlin"]))
    return chosen


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


def tolerance(sd_a, sd_b):
    """Two combined SDs, but never tighter than 0.5 percentage points."""
    return max(2 * math.sqrt(sd_a ** 2 + sd_b ** 2), 0.5)


def verdict(c, paper):
    """Is the gap from the paper's mean within run-to-run noise?"""
    p_mean, p_sd = paper["acc"]
    diff = c["acc"] - p_mean
    tol = tolerance(c["acc_sd"], p_sd)
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
    keys = sorted(best, key=lambda k: (k[0], SAMPLING_ORDER.index(k[1]), nonlin_rank(k[2])))
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
                "ca": "Ca-only", "v1": "V1 mixed", "gate": "Ca gate (BAC)"}


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
                line_label = nonlin_label(cs[0].get("variant", n))  # shows tuned constants
                ax.plot(px, m, color=NONLIN_COLOR[n], marker=NONLIN_STYLE[n]["marker"],
                        mec=SURFACE, mew=1.2, lw=2.6 if n in ("v1", "gate") else 2,
                        label=line_label, zorder=3)
                ends.append((px[-1], m[-1], line_label))
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
            "V1 mixed line, mixing adds nothing. Single-nonlinearity models and the Ca gate have "
            "the baseline's parameter count; V1 mixed has three cable weights per dendrite. "
            "Error bars and shading: SD over seeds. Constants in [brackets] were chosen on "
            "validation accuracy.")


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
        tol = tolerance(v["acc_sd"], b["acc_sd"])
        row = {"dataset": ds, "model": SAMPLING_NAME[s], "size": f"{dends} dends x {soma} somata",
               "seeds": min(c["n"] for c in group.values())}
        for n in NONLIN_ORDER:
            row[f"{n}_acc"] = round(group[n]["acc"], 3) if n in group else ""
            row[f"{n}_acc_sd"] = round(group[n]["acc_sd"], 3) if n in group else ""
        row.update(best_single=best_n, gap_pp=round(gap, 3), tolerance_pp=round(tol, 3),
                   verdict="mixed wins" if gap > tol else "single is enough")
        if "gate" in group:  # the Ca gate is judged against the same best single nonlinearity
            gg = group["gate"]
            g_gap, g_tol = gg["acc"] - b["acc"], tolerance(gg["acc_sd"], b["acc_sd"])
            row.update(gate_gap_pp=round(g_gap, 3), gate_tolerance_pp=round(g_tol, 3),
                       gate_verdict="gate wins" if g_gap > g_tol else "no advantage")
        else:
            row.update(gate_gap_pp="", gate_tolerance_pp="", gate_verdict="")
        row["variants"] = "; ".join(f"{n}={group[n].get('variant', n)}" for n in NONLIN_ORDER
                                    if n in group and group[n].get("variant", n) != n)
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
        if r["gate_verdict"]:
            print(f"{'':9s} {'':9s} {'':22s} {'':5s}{'Ca gate vs best single':>{18 * len(NONLIN_ORDER)}s}  "
                  f"{'':>11s} {r['gate_gap_pp']:+6.2f}  {r['gate_verdict']}")
        if r["variants"]:
            print(f"{'':9s} tuned variants shown: {r['variants']}")
    if any(r["seeds"] < 3 for r in rows):
        print("  note: fewer than 3 seeds, so the SD and the verdict are unreliable; "
              "the tolerance floor of 0.5 pp is doing the work.")
    return rows


# ---------------------------------------------------------------------------
# Constants sweep: variants of one model that differ only in the fixed constants
# ---------------------------------------------------------------------------

def _variants_by_base(configs):
    """(dataset, sampling, dends, soma, base) -> [configs], only where a tuned variant exists."""
    groups = defaultdict(list)
    for c in configs:
        groups[(c["dataset"], c["sampling"], c["dends"], c["soma"], base_of(c["nonlin"]))].append(c)
    keep = {k: v for k, v in groups.items() if any(c["nonlin"] != k[4] for c in v)}
    return sorted(keep.items(), key=lambda kv: (kv[0][0], SAMPLING_ORDER.index(kv[0][1]),
                                                kv[0][3], kv[0][2], nonlin_rank(kv[0][4])))


def grid_line(c):
    """'ca|--ca-thresh 0.5 --leak 0.1': a sweep_constants.sh line that reproduces this variant."""
    base = base_of(c["nonlin"])
    head = f"branch --mix {branch_mix(base)}" if is_branch(base) else base
    flags = []
    k = c["constants"]
    if base == "gate" and k.get("basal_nonlin", "leaky") != "leaky":
        flags.append(f"--basal-nonlin {k['basal_nonlin']}")
    for key, default in CONSTANT_DEFAULTS.items():
        if base == "leaky" or (key == "apical_frac" and base != "gate"):
            continue
        if float(k.get(key, default)) != default:
            flags.append(f"--{key.replace('_', '-')} {float(k[key]):g}")
    if k.get("weight_decay"):
        flags.append(f"--weight-decay {float(k['weight_decay']):g}")
    if k.get("apical_input", "same") != "same":
        head += f" --apical-input {k['apical_input']}"
    return f"{head}|{' '.join(flags)}"


def write_constants_sweep(configs, out):
    """constants_sweep.csv: every variant of each tuned model, ranked by validation accuracy."""
    rows, winners = [], []
    for (ds, s, dends, soma, base), cs in _variants_by_base(configs):
        score = dict(zip((c["nonlin"] for c in cs), selection_scores(cs)))
        ranked = sorted(cs, key=lambda c: -score[c["nonlin"]])
        winners.append(ranked[0])
        for i, c in enumerate(ranked):
            rows.append({"dataset": ds, "model": SAMPLING_NAME[s],
                         "size": f"{dends} dends x {soma} somata", "base": nonlin_label(base),
                         "variant": variant_suffix(c["nonlin"]) or "(defaults)", "seeds": c["n"],
                         "val_matched": round(score[c["nonlin"]], 3),
                         "val_acc": round(c["val_acc"], 3), "val_acc_sd": round(c["val_acc_sd"], 3),
                         "test_acc": round(c["acc"], 3), "test_acc_sd": round(c["acc_sd"], 3),
                         "selected": "yes" if i == 0 else "", "grid_line": grid_line(c)})
    if not rows:
        print("  skipping constants_sweep.csv: no runs with non-default constants")
        return rows
    path = out / "constants_sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")

    print("\nConstants sweep: variants ranked by VALIDATION accuracy (%) on the seeds they share "
          "('matched'); test shown for reference")
    print(f"{'dataset':9s} {'size':22s} {'model':20s} {'variant':26s} {'seeds':>5s}  "
          f"{'matched':>7s}  {'val acc':>14s}  {'test acc':>14s}  pick")
    for r in rows:
        print(f"{r['dataset']:9s} {r['size']:22s} {r['base']:20s} {r['variant']:26s} "
              f"{r['seeds']:>5d}  {r['val_matched']:7.2f}  {r['val_acc']:6.2f} +/- {r['val_acc_sd']:4.2f}  "
              f"{r['test_acc']:6.2f} +/- {r['test_acc_sd']:4.2f}  {'<-' if r['selected'] else ''}")
    if any(r["seeds"] < 2 for r in rows):
        print("  note: single-seed variants; differences under ~1 pp are noise. Rerun the "
              "winners with more seeds (stage 2) before drawing conclusions.")
    print("\nStage-2 grid lines (winners), for sweep_constants.sh via GRID=<file>:")
    for c in winners:
        print(f"  {grid_line(c)}")
    return rows


def fig_constants_sweep(configs, datasets, out):
    """Validation accuracy per variant for each tuned model, one panel per model and size."""
    panels = [(k, cs) for k, cs in _variants_by_base(configs) if k[0] in datasets]
    if not panels:
        print("  skipping 9_constants_sweep.png: no runs with non-default constants")
        return
    ncol = 2 if len(panels) > 1 else 1
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, ((ds, s, dends, soma, base), cs) in zip(axes.flat, panels):
        cs = sorted(cs, key=lambda c: nonlin_rank(c["nonlin"]))
        x = np.arange(len(cs))
        color = nonlin_color(base)
        val = np.array([c["val_acc"] for c in cs]); vsd = np.array([c["val_acc_sd"] for c in cs])
        test = np.array([c["acc"] for c in cs])
        ax.errorbar(x, val, yerr=vsd, fmt="o", ms=7, mfc=color, mec=SURFACE, mew=1.2,
                    ecolor=color, elinewidth=1.3, capsize=3, label="validation", zorder=4)
        ax.plot(x, test, "o", ms=7, mfc="none", mec=color, mew=1.3, label="test", zorder=3)
        best = int(np.argmax(selection_scores(cs)))
        ax.annotate("selected", (x[best], val[best]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([variant_suffix(c["nonlin"]) or "defaults" for c in cs],
                           rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(cs) - 0.4)
        ax.margins(y=0.3)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {nonlin_label(base)}, {dends} x {soma}")
    axes.flat[0].legend(loc="lower right")
    _finish(fig, out / "9_constants_sweep.png",
            "Each panel is one model whose fixed constants were varied; x labels are the "
            "non-default settings. Filled: validation accuracy, hollow: test. Error bars: SD over "
            "seeds (none for single-seed runs). 'selected' is the best validation accuracy on "
            "the seed every variant was trained with.")


# ---------------------------------------------------------------------------
# Paired comparison with the baseline (same seed = same data split and wiring)
# ---------------------------------------------------------------------------

def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t, by integrating the density (no SciPy needed)."""
    if df < 1 or not math.isfinite(t):
        return float("nan")
    x = np.linspace(0.0, abs(t), 20001)
    logc = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(logc - (df + 1) / 2 * np.log1p(x ** 2 / df))
    area = float(np.sum((pdf[1:] + pdf[:-1]) * np.diff(x)) / 2)  # P(0 < T < |t|)
    return max(0.0, 1.0 - 2.0 * area)


def paired_stats(a, b):
    """Differences a - b over the seeds both dicts {seed: value} share."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in seeds], dtype=float)
    n = len(d)
    if n == 0:
        return None
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    return dict(n=n, seeds=seeds, diffs=d, mean=mean, sd=sd, t=t,
                p=t_two_sided_p(t, n - 1) if n > 1 else float("nan"), wins=int((d > 0).sum()))


def _same_decay_baseline(c, by_variant):
    """The LeakyReLU run this model should be paired with: same weight decay."""
    wd = c["constants"].get("weight_decay", 0.0)
    return by_variant.get(f"leaky@wd{wd:g}" if wd else "leaky")


def write_paired(configs, selected, out):
    """paired_vs_baseline.csv: each model minus the LeakyReLU baseline, seed by seed.

    Runs with the same seed share the data split and the synaptic wiring, so the paired
    difference removes seed-to-seed variation that the unpaired SDs include. Lists the
    selected variant of every model plus any other variant with at least two shared seeds
    (so controls such as a gain sweep all appear). Models trained with weight decay are
    paired with the baseline trained with the same weight decay.
    """
    all_by_size = defaultdict(dict)
    for c in configs:
        all_by_size[(c["dataset"], c["sampling"], c["dends"], c["soma"])][c["nonlin"]] = c
    rows = []
    for key, sel in _size_groups(selected):
        by_variant = all_by_size[key]
        picked = {c["variant"] for c in sel.values()}
        for v in sorted(by_variant, key=nonlin_rank):
            c = by_variant[v]
            base = _same_decay_baseline(c, by_variant)
            if base is None or base is c:
                continue
            st = paired_stats(c["acc_by_seed"], base["acc_by_seed"])
            if st is None or (v not in picked and st["n"] < 2):
                continue
            best = paired_stats(c["acc_best_by_seed"], base["acc_best_by_seed"])
            ds, s, dends, soma = key
            rows.append({
                "dataset": ds, "model": SAMPLING_NAME[s], "size": f"{dends} dends x {soma} somata",
                "variant": nonlin_label(v), "baseline": nonlin_label(base["nonlin"]),
                "seeds": st["n"], "acc": round(float(np.mean([c["acc_by_seed"][k] for k in st["seeds"]])), 3),
                "baseline_acc": round(float(np.mean([base["acc_by_seed"][k] for k in st["seeds"]])), 3),
                "diff": round(st["mean"], 3), "diff_sd": round(st["sd"], 3),
                "t": round(st["t"], 3) if math.isfinite(st["t"]) else "",
                "p": round(st["p"], 4) if math.isfinite(st["p"]) else "",
                "wins": f"{st['wins']}/{st['n']}", "selected": "yes" if v in picked else "",
                "diff_early_stop": round(best["mean"], 3) if best else "",
                "seeds_early_stop": best["n"] if best else 0,
            })
    if not rows:
        print("  skipping paired_vs_baseline.csv: no model shares seeds with the baseline")
        return rows
    path = out / "paired_vs_baseline.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")

    wv = max(len(r["variant"]) for r in rows)
    wb = max(len(r["baseline"]) for r in rows)
    print("\nPaired with the baseline on shared seeds: test accuracy difference (pp)")
    print(f"{'dataset':9s} {'size':22s} {'model':{wv}s} {'vs':{wb}s} {'n':>2s}  {'acc':>6s} "
          f"{'base':>6s}  {'diff':>14s}  {'t':>6s} {'p':>7s}  {'wins':>4s} {'pick':>4s}  early-stop diff")
    for r in rows:
        t = f"{r['t']:6.2f}" if r["t"] != "" else f"{'':6s}"
        pv = f"{r['p']:7.4f}" if r["p"] != "" else f"{'':7s}"
        es = (f"{r['diff_early_stop']:+.2f} (n={r['seeds_early_stop']})"
              if r["diff_early_stop"] != "" else "n/a")
        print(f"{r['dataset']:9s} {r['size']:22s} {r['variant']:{wv}s} {r['baseline']:{wb}s} "
              f"{r['seeds']:>2d}  {r['acc']:6.2f} {r['baseline_acc']:6.2f}  "
              f"{r['diff']:+6.2f} +/- {r['diff_sd']:4.2f}  {t} {pv}  {r['wins']:>4s} "
              f"{'<-' if r['selected'] else '':>4s}  {es}")
    print("  note: p is a two-sided paired t-test over seeds, uncorrected for the number of "
          "models compared; with few seeds treat it as suggestive. 'early-stop diff' uses test "
          "accuracy at each run's best-validation epoch (n/a for runs made before it was saved).")
    return rows


def fig_size_generalisation(configs, selected, datasets, out):
    """Does a model's edge over the baseline hold at other model sizes?"""
    plain = {(c["dataset"], c["sampling"], c["dends"], c["soma"]): c
             for c in configs if c["nonlin"] == "leaky"}
    # One variant per model, the same at every size: the one trained at the most sizes
    # (ties go to the variant --select picked more often), so the line compares like with like.
    picked = defaultdict(int)
    for c in selected:
        picked[(c["dataset"], c["sampling"], c["variant"])] += 1
    by_variant = defaultdict(list)  # (dataset, sampling, variant) -> [(size key, config, baseline)]
    for c in configs:
        key = (c["dataset"], c["sampling"], c["dends"], c["soma"])
        if (base_of(c["nonlin"]) == "leaky" or key not in plain or c["dataset"] not in datasets
                or c["constants"].get("weight_decay")):
            continue  # decayed models are compared in the paired table, not here
        by_variant[(c["dataset"], c["sampling"], c["nonlin"])].append((key, c, plain[key]))
    lines = {}  # (dataset, sampling, base) -> [(size key, config, baseline)]
    for (ds, s, v), pts in by_variant.items():
        k = (ds, s, base_of(v))
        rank = (len(pts), picked[(ds, s, v)])
        if k not in lines or rank > lines[k][0]:
            lines[k] = (rank, pts)
    lines = {k: pts for k, (_, pts) in lines.items()}
    lines = {k: sorted(v, key=lambda t: t[1]["n_params"]) for k, v in lines.items() if len(v) > 1}
    if not lines:
        print("  skipping 11_size_generalisation.png: no tuned model at two or more sizes")
        return
    rows = sorted({(ds, s) for ds, s, _ in lines},
                  key=lambda k: (k[0], SAMPLING_ORDER.index(k[1])))
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.8 * len(rows)), squeeze=False)
    for r, (ds, s) in enumerate(rows):
        ax, axd = axes[r]
        bases = sorted((b for d2, s2, b in lines if (d2, s2) == (ds, s)), key=nonlin_rank)
        sizes = sorted({t[0] for b in bases for t in lines[(ds, s, b)]},
                       key=lambda k: plain[k]["n_params"])
        xs = {k: i for i, k in enumerate(sizes)}
        base_cs = [plain[k] for k in sizes]
        ax.errorbar([c["n_params"] for c in base_cs], [c["acc"] for c in base_cs],
                    yerr=[c["acc_sd"] for c in base_cs], color=NONLIN_COLOR["leaky"],
                    mec=SURFACE, mew=1.2, capsize=3, elinewidth=1.2,
                    label=nonlin_label("leaky"), **NONLIN_STYLE["leaky"])
        width = 0.8 / len(bases)
        for j, b in enumerate(bases):
            pts = lines[(ds, s, b)]
            color = nonlin_color(b)
            label = nonlin_label(pts[0][1]["nonlin"])
            ax.errorbar([c["n_params"] for _, c, _ in pts], [c["acc"] for _, c, _ in pts],
                        yerr=[c["acc_sd"] for _, c, _ in pts], color=color, mec=SURFACE,
                        mew=1.2, capsize=3, elinewidth=1.2, label=label, **nonlin_style(b))
            for k, c, base in pts:
                st = paired_stats(c["acc_by_seed"], base["acc_by_seed"])
                if st is None:
                    continue
                x = xs[k] + (j - (len(bases) - 1) / 2) * width
                axd.errorbar(x, st["mean"], yerr=st["sd"], fmt=nonlin_style(b)["marker"], ms=8,
                             mfc=color, mec=SURFACE, mew=1.2, ecolor=color, elinewidth=1.3,
                             capsize=4, zorder=4)
                axd.plot(np.full(st["n"], x + 0.12), st["diffs"], ".", ms=5, color=color,
                         alpha=0.55, zorder=3)
                axd.annotate(f"{st['mean']:+.2f}\n{st['wins']}/{st['n']} seeds", (x, st["mean"]),
                             xytext=(-10, 0), textcoords="offset points", ha="right",
                             va="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis)
        ax.margins(x=0.12)
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {SAMPLING_NAME[s]}: accuracy by model size")
        ax.legend(loc="lower right")
        axd.axhline(0, color=MUTED, lw=1)
        axd.set_xticks(range(len(sizes)))
        axd.set_xticklabels([f"{k[2]} x {k[3]}" for k in sizes])
        axd.set_xlim(-0.7, len(sizes) - 0.4)
        axd.grid(axis="x", visible=False)
        axd.margins(y=0.25)
        axd.set_xlabel("dendrites x somata")
        axd.set_ylabel("Accuracy minus baseline (pp)")
        axd.set_title("Paired difference, same seeds")
    _finish(fig, out / "11_size_generalisation.png",
            "Left: mean and SD over seeds at each model size. Right: each model minus the "
            "LeakyReLU baseline trained with the same seed (large marker: mean and SD of the "
            "differences; small dots: individual seeds). Above zero means better than the baseline.")


# ---------------------------------------------------------------------------
# Ratio experiment: per-dendrite nonlinearity mixes (--nonlin branch --mix S:B:C)
# ---------------------------------------------------------------------------

TRAIN_SIZE = {"fmnist": 54000, "cifar10": 45000}  # training images after the 10% validation split


def has_context(c):
    return c["constants"].get("apical_input", "same") != "same"


def _subset_curves(configs):
    """{(dataset, sampling, dends, soma): {variant: {n_train: config}}} for --train-subset runs.

    Subset runs must match the paper's other settings; the full-data run of the same variant
    (if any) is added as the last point. Only groups that include the baseline are returned.
    """
    other = {k: v for k, v in PAPER_SETTINGS.items() if k != "train_subset"}
    ok = [c for c in configs if all(c[k] == v for k, v in other.items()) and not has_context(c)]
    curves = defaultdict(lambda: defaultdict(dict))
    for c in ok:
        if c["train_subset"]:
            key = (c["dataset"], c["sampling"], c["dends"], c["soma"])
            pts = curves[key][c["nonlin"]]
            # several epoch settings for one subset size: keep the one with more seeds
            if c["train_subset"] not in pts or c["n"] > pts[c["train_subset"]]["n"]:
                pts[c["train_subset"]] = c
    for c in ok:
        key = (c["dataset"], c["sampling"], c["dends"], c["soma"])
        if not c["train_subset"] and key in curves and c["nonlin"] in curves[key]:
            curves[key][c["nonlin"]][TRAIN_SIZE[c["dataset"]]] = c
    return {k: dict(v) for k, v in curves.items() if "leaky" in v}


def _efficiency_metric(by_variant, n_train):
    """'acc_best_by_seed' when every run at this training-set size recorded it, else final."""
    cs = [pts[n_train] for pts in by_variant.values() if n_train in pts]
    early = all(set(c["acc_best_by_seed"]) == set(c["acc_by_seed"]) for c in cs)
    return "acc_best_by_seed" if early else "acc_by_seed"


def fig_data_efficiency(configs, datasets, out):
    """14_data_efficiency.png + data_efficiency.csv: accuracy against training-set size."""
    curves = {k: v for k, v in _subset_curves(configs).items() if k[0] in datasets}
    if not curves:
        print("  skipping 14_data_efficiency.png: no --train-subset runs with a baseline")
        return []
    keys = sorted(curves, key=lambda k: (k[0], SAMPLING_ORDER.index(k[1]), k[2] * k[3]))
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 3.8 * len(keys)), squeeze=False)
    rows = []
    for r, key in enumerate(keys):
        ds, s, dends, soma = key
        by_variant = curves[key]
        ax, axd = axes[r]
        sizes = sorted({n for pts in by_variant.values() for n in pts})
        metric = {n: _efficiency_metric(by_variant, n) for n in sizes}
        variants = sorted(by_variant, key=nonlin_rank)
        others = [v for v in variants if v != "leaky"]
        for j, v in enumerate(variants):
            pts = by_variant[v]
            ns = sorted(pts)
            stats = [mean_sd(list(pts[n][metric[n]].values())) for n in ns]
            b = base_of(v)
            ax.errorbar(ns, [m for m, _ in stats], yerr=[sd for _, sd in stats],
                        color=nonlin_color(b), mec=SURFACE, mew=1.2, capsize=3, elinewidth=1.2,
                        label=nonlin_label(v), **nonlin_style(b))
            for n, (m, sd) in zip(ns, stats):
                st = (paired_stats(pts[n][metric[n]], by_variant["leaky"][n][metric[n]])
                      if v != "leaky" and n in by_variant["leaky"] else None)
                rows.append({
                    "dataset": ds, "model": SAMPLING_NAME[s], "size": f"{dends} dends x {soma} somata",
                    "variant": nonlin_label(v), "train_images": n, "epochs": pts[n]["epochs"],
                    "seeds": pts[n]["n"], "acc": round(m, 3), "acc_sd": round(sd, 3),
                    "metric": "early-stop" if metric[n] == "acc_best_by_seed" else "final epoch",
                    "diff": round(st["mean"], 3) if st else "",
                    "diff_sd": round(st["sd"], 3) if st else "",
                    "p": round(st["p"], 4) if st and math.isfinite(st["p"]) else "",
                    "wins": f"{st['wins']}/{st['n']}" if st else "",
                })
                if st is None:
                    continue
                k = others.index(v)
                x = sizes.index(n) + (k - (len(others) - 1) / 2) * 0.8 / max(len(others), 1)
                axd.errorbar(x, st["mean"], yerr=st["sd"], fmt=nonlin_style(b)["marker"], ms=8,
                             mfc=nonlin_color(b), mec=SURFACE, mew=1.2, ecolor=nonlin_color(b),
                             elinewidth=1.3, capsize=4, zorder=4)
                axd.plot(np.full(st["n"], x + 0.1), st["diffs"], ".", ms=5,
                         color=nonlin_color(b), alpha=0.55, zorder=3)
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
        ax.minorticks_off()
        ax.margins(x=0.08)
        ax.set_xlabel("Training images (log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {SAMPLING_NAME[s]} {dends} x {soma}: "
                     "accuracy by training-set size")
        ax.legend(loc="lower right")
        axd.axhline(0, color=MUTED, lw=1)
        axd.set_xticks(range(len(sizes)))
        axd.set_xticklabels([f"{n:,}" for n in sizes])
        axd.set_xlim(-0.6, len(sizes) - 0.4)
        axd.grid(axis="x", visible=False)
        axd.margins(y=0.25)
        axd.set_xlabel("Training images")
        axd.set_ylabel("Accuracy minus baseline (pp)")
        axd.set_title("Paired difference, same seeds")
    _finish(fig, out / "14_data_efficiency.png",
            "Left: mean and SD over seeds when training on a subset of the images (smaller "
            "subsets train for more epochs; test accuracy is taken at the best-validation epoch "
            "where every run recorded it). Right: each model minus the LeakyReLU baseline "
            "trained on the same images with the same seed.")
    path = out / "data_efficiency.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")
    print("\nData efficiency: test accuracy (%) by number of training images")
    wv = max(len(r["variant"]) for r in rows)
    for r in rows:
        d = (f"{r['diff']:+6.2f} +/- {r['diff_sd']:4.2f}  p {r['p']}  wins {r['wins']}"
             if r["diff"] != "" else "")
        print(f"{r['dataset']:9s} {r['size']:22s} {r['train_images']:>6,d} images  "
              f"{r['variant']:{wv}s} {r['seeds']:>2d}  {r['acc']:6.2f} +/- {r['acc_sd']:4.2f}  "
              f"({r['metric']})  {d}")
    return rows


CONTEXT_CELLS = (("leaky", "additive soma, apical sees local patches (paper)"),
                 ("leaky@apcoarse", "additive soma, apical sees coarse context"),
                 ("gate", "Ca gate, apical sees local patches"),
                 ("gate@apcoarse", "Ca gate, apical sees coarse context"))
# Steep saturating dendrites combined with coarse apical input: do the two gains add?
STACK_TYPES = (("ca@thr0", "Ca-only (threshold 0)"), ("sub@sat0.2", "matched tanh"))
STACK_KEYS = {k for v, _ in STACK_TYPES for k in (v, v + "_apcoarse")}


def write_apical_context(configs, datasets, out):
    """apical_context.csv + 15_apical_context.png: the 2 x 2 of soma type by apical input.

    The gate hypothesis needs more than 'coarse context helps': the gate must gain more from
    context than an additive soma does. That is the interaction contrast, paired by seed.
    """
    groups = defaultdict(dict)
    for c in configs:
        if c["dataset"] in datasets and (c["nonlin"] in dict(CONTEXT_CELLS) or c["nonlin"] in STACK_KEYS):
            groups[(c["dataset"], c["sampling"], c["dends"], c["soma"])][c["nonlin"]] = c
    groups = {k: g for k, g in groups.items() if any(has_context(c) for c in g.values())}
    if not groups:
        print("  skipping apical_context.csv: no --apical-input coarse runs")
        return []
    rows = []
    stacked = set()
    keys = sorted(groups, key=lambda k: (k[0], SAMPLING_ORDER.index(k[1]), k[2] * k[3]))
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.9), squeeze=False)
    for ax, key in zip(axes[0], keys):
        ds, s, dends, soma = key
        g = groups[key]
        where = dict(dataset=ds, model=SAMPLING_NAME[s], size=f"{dends} dends x {soma} somata")

        def contrast(name, a, b):
            if a not in g or b not in g:
                return
            st = paired_stats(g[a]["acc_by_seed"], g[b]["acc_by_seed"])
            if st:
                rows.append(dict(where, row=name, seeds=st["n"], acc="", acc_sd="",
                                 diff=round(st["mean"], 3), diff_sd=round(st["sd"], 3),
                                 p=round(st["p"], 4) if math.isfinite(st["p"]) else "",
                                 wins=f"{st['wins']}/{st['n']}"))

        for v, label in CONTEXT_CELLS:
            if v in g:
                rows.append(dict(where, row=label, seeds=g[v]["n"], acc=round(g[v]["acc"], 3),
                                 acc_sd=round(g[v]["acc_sd"], 3), diff="", diff_sd="", p="", wins=""))
        contrast("context effect, additive soma", "leaky@apcoarse", "leaky")
        contrast("context effect, Ca gate", "gate@apcoarse", "gate")
        contrast("gate effect, local apical input", "gate", "leaky")
        contrast("gate effect, coarse apical input", "gate@apcoarse", "leaky@apcoarse")
        if all(v in g for v, _ in CONTEXT_CELLS):  # (gate: coarse - local) - (additive: coarse - local)
            seeds = set.intersection(*(set(g[v]["acc_by_seed"]) for v, _ in CONTEXT_CELLS))
            a = {k: g["gate@apcoarse"]["acc_by_seed"][k] - g["gate"]["acc_by_seed"][k] for k in seeds}
            b = {k: g["leaky@apcoarse"]["acc_by_seed"][k] - g["leaky"]["acc_by_seed"][k] for k in seeds}
            st = paired_stats(a, b)
            if st:
                rows.append(dict(where, row="interaction: does the gate gain more from context?",
                                 seeds=st["n"], acc="", acc_sd="", diff=round(st["mean"], 3),
                                 diff_sd=round(st["sd"], 3),
                                 p=round(st["p"], 4) if math.isfinite(st["p"]) else "",
                                 wins=f"{st['wins']}/{st['n']}"))
        lines = [("leaky", "leaky@apcoarse", NONLIN_COLOR["leaky"], "additive soma", "leaky", -0.07),
                 ("gate", "gate@apcoarse", NONLIN_COLOR["gate"], "Ca gate", "gate", 0.07)]
        for j, (v, label) in enumerate(STACK_TYPES):
            both = v + "_apcoarse"
            if both not in g:
                continue
            stacked.add(key)
            base = v.split("@")[0]
            lines.append((v, both, NONLIN_COLOR[base], f"{label} dendrites", base, 0.14 + 0.07 * j))
            for k, where_txt in ((v, "local patches"), (both, "coarse context")):
                if k in g:
                    rows.append(dict(where, row=f"{label} dendrites, apical sees {where_txt}",
                                     seeds=g[k]["n"], acc=round(g[k]["acc"], 3),
                                     acc_sd=round(g[k]["acc_sd"], 3), diff="", diff_sd="", p="", wins=""))
            contrast(f"context effect, {label} dendrites", both, v)
            contrast(f"{label} effect, coarse apical input", both, "leaky@apcoarse")
            contrast(f"{label} + coarse context vs the paper's model", both, "leaky")
            if all(k in g for k in (v, both, "leaky", "leaky@apcoarse")):
                seeds = set.intersection(*(set(g[k]["acc_by_seed"])
                                           for k in (v, both, "leaky", "leaky@apcoarse")))
                a = {k: g[both]["acc_by_seed"][k] - g[v]["acc_by_seed"][k] for k in seeds}
                b = {k: g["leaky@apcoarse"]["acc_by_seed"][k] - g["leaky"]["acc_by_seed"][k] for k in seeds}
                st = paired_stats(a, b)
                if st:
                    rows.append(dict(where, row=f"interaction: {label} x context (0 = the gains add)",
                                     seeds=st["n"], acc="", acc_sd="", diff=round(st["mean"], 3),
                                     diff_sd=round(st["sd"], 3),
                                     p=round(st["p"], 4) if math.isfinite(st["p"]) else "",
                                     wins=f"{st['wins']}/{st['n']}"))
        for local, coarse, color, name, style, dx in lines:
            cells = [(i, g[v]) for i, v in enumerate((local, coarse)) if v in g]
            if not cells:
                continue
            ax.errorbar([i for i, _ in cells], [c["acc"] for _, c in cells],
                        yerr=[c["acc_sd"] for _, c in cells], color=color, mec=SURFACE, mew=1.2,
                        capsize=4, elinewidth=1.3, label=name, **NONLIN_STYLE[style])
            for i, c in cells:
                ax.plot(np.full(c["n"], i + dx),
                        list(c["acc_by_seed"].values()), ".", ms=5, color=color, alpha=0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["local patches\n(same as basal)", "coarse context\n(pooled image)"])
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.2)
        if key in stacked:  # four lines: leave room for the legend above the data
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        ax.set_xlabel("What the apical dendrites see")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {dends} x {soma}")
        ax.legend(loc="best")
    _finish(fig, out / "15_apical_context.png",
            "Mean and SD over seeds (dots: individual seeds). Parallel lines mean context helps "
            "any soma equally; the gate line rising more steeply would mean the multiplicative "
            "gate makes specific use of context."
            + (" Ca-only and matched tanh lines: steep saturating dendrites with an additive soma; "
               "parallel to the additive line means the two gains add." if stacked else ""))
    path = out / "apical_context.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")
    print("\nApical context, 2 x 2 (soma type by apical input): test accuracy (%) and paired contrasts (pp)")
    wr = max(len(r["row"]) for r in rows)
    for r in rows:
        if r["acc"] != "":
            val = f"{r['acc']:6.2f} +/- {r['acc_sd']:4.2f}"
        else:
            val = f"{r['diff']:+6.2f} +/- {r['diff_sd']:4.2f}  p {r['p']}  wins {r['wins']}"
        print(f"{r['dataset']:9s} {r['size']:22s} {r['row']:{wr}s} {r['seeds']:>2d}  {val}")
    return rows


def _size_groups(configs):
    """(dataset, sampling, dends, soma) -> {variant: config}, sorted."""
    groups = defaultdict(dict)
    for c in configs:
        groups[(c["dataset"], c["sampling"], c["dends"], c["soma"])][c["nonlin"]] = c
    return sorted(groups.items(), key=lambda kv: (kv[0][0], SAMPLING_ORDER.index(kv[0][1]),
                                                  kv[0][3], kv[0][2]))


def _ratio_references(group):
    """Reference models at one size: the best single nonlinearity and V1 mixed, if present."""
    refs = []
    singles = [n for n in SINGLE_NONLINS if n in group]
    if singles:
        best = max(singles, key=lambda n: group[n]["acc"])
        refs.append((best, f"best single ({nonlin_label(best)})", NONLIN_COLOR[best], "-"))
    if "v1" in group:
        refs.append(("v1", NONLIN_LABEL["v1"], NONLIN_COLOR["v1"], (0, (2, 1.5))))
    return refs


def fig_ratio_sweep(configs, datasets, out):
    """Per-dendrite ratio models vs the best single nonlinearity and V1 mixed, per size."""
    panels = [(key, g) for key, g in _size_groups(configs)
              if key[0] in datasets and any(is_branch(n) for n in g)]
    if not panels:
        print("  skipping 8_ratio_sweep.png: no --nonlin branch runs")
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.8 * len(panels)), squeeze=False)
    for row, ((ds, s, dends, soma), g) in enumerate(panels):
        mixes = sorted((n for n in g if is_branch(n)), key=nonlin_rank)
        for col, (metric, label) in enumerate([("acc", "Test accuracy (%)"),
                                               ("loss", "Test loss")]):
            ax = axes[row, col]
            for n, name, color, ls in _ratio_references(g):
                c = g[n]
                m, sd = c[metric], c[f"{metric}_sd"]
                ax.axhspan(m - sd, m + sd, color=color, alpha=0.08, lw=0)
                ax.axhline(m, color=color, lw=1.3, ls=ls, label=name)
            fmt = "{:.2f}" if metric == "acc" else "{:.3f}"
            for i, n in enumerate(mixes):
                c = g[n]
                ax.errorbar(i, c[metric], yerr=c[f"{metric}_sd"], fmt=BRANCH_STYLE["marker"],
                            ms=9, mfc=nonlin_color(n), mec=SURFACE, mew=1.2,
                            ecolor=nonlin_color(n), elinewidth=1.4, capsize=4, zorder=4)
                ax.annotate(fmt.format(c[metric]), (i, c[metric]), xytext=(10, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
            ax.set_xticks(np.arange(len(mixes)))
            ax.set_xticklabels([branch_mix(n) for n in mixes])
            ax.set_xlim(-0.6, len(mixes) - 0.4)
            ax.set_xlabel("supra : sub : Ca dendrites per soma")
            ax.set_ylabel(label)
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.3)
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}, {SAMPLING_NAME[s]}, {dends} x {soma}: "
                         f"ratio sweep, test {'accuracy' if metric == 'acc' else 'loss'}")
            if row == 0 and col == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower left")
    _finish(fig, out / "8_ratio_sweep.png",
            "Each point is one per-dendrite ratio (every dendrite computes one nonlinearity; the "
            "ratio sets how many of a soma's dendrites are supra, sub and Ca). Lines and bands: "
            "the best single-nonlinearity model and the V1 mixed model at the same size, mean "
            "and SD over seeds. Ratio models have the baseline parameter count.")


def write_ratio_sweep(configs, out):
    """ratio_sweep.csv: each ratio vs the best single nonlinearity and V1 mixed at its size."""
    rows = []
    for (ds, s, dends, soma), g in _size_groups(configs):
        mixes = sorted((n for n in g if is_branch(n)), key=nonlin_rank)
        if not mixes:
            continue
        singles = [n for n in SINGLE_NONLINS if n in g]
        best = max(singles, key=lambda n: g[n]["acc"]) if singles else None
        for n in mixes:
            c = g[n]
            row = {"dataset": ds, "model": SAMPLING_NAME[s], "size": f"{dends} dends x {soma} somata",
                   "mix": branch_mix(n), "seeds": c["n"], "acc": round(c["acc"], 3),
                   "acc_sd": round(c["acc_sd"], 3), "loss": round(c["loss"], 4),
                   "best_single": best or "", "best_single_acc": "", "gap_vs_best_single_pp": "",
                   "v1_acc": "", "gap_vs_v1_pp": "", "tolerance_pp": "", "verdict": ""}
            if best:
                b = g[best]
                gap, tol = c["acc"] - b["acc"], tolerance(c["acc_sd"], b["acc_sd"])
                row.update(best_single_acc=round(b["acc"], 3), gap_vs_best_single_pp=round(gap, 3),
                           tolerance_pp=round(tol, 3),
                           verdict="ratio wins" if gap > tol else "no advantage")
            if "v1" in g:
                row.update(v1_acc=round(g["v1"]["acc"], 3),
                           gap_vs_v1_pp=round(c["acc"] - g["v1"]["acc"], 3))
            rows.append(row)
    if not rows:
        print("  skipping ratio_sweep.csv: no --nonlin branch runs")
        return rows
    path = out / "ratio_sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")

    print("\nRatio experiment: per-dendrite supra:sub:Ca mixes, test accuracy (%)")
    print(f"{'dataset':9s} {'model':9s} {'size':22s} {'mix':>8s} {'seeds':>5s}  {'acc':>14s}  "
          f"{'best single':>22s}  {'vs best':>7s}  {'vs V1':>7s}  verdict")
    for r in rows:
        acc = f"{r['acc']:.2f} +/- {r['acc_sd']:.2f}"
        bs = f"{r['best_single']} {r['best_single_acc']:.2f}" if r["best_single"] else "-"
        vb = f"{r['gap_vs_best_single_pp']:+.2f}" if r["best_single"] else "-"
        vv = f"{r['gap_vs_v1_pp']:+.2f}" if r["v1_acc"] != "" else "-"
        print(f"{r['dataset']:9s} {r['model']:9s} {r['size']:22s} {r['mix']:>8s} {r['seeds']:>5d}  "
              f"{acc:>14s}  {bs:>22s}  {vb:>7s}  {vv:>7s}  {r['verdict']}")
    if any(r["seeds"] < 3 for r in rows):
        print("  note: fewer than 3 seeds, so the SD and the verdict are unreliable.")
    return rows


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary(best, n_configs, out):
    rows = []
    order = lambda kv: (kv[0][0], SAMPLING_ORDER.index(kv[0][1]), nonlin_rank(kv[0][2]))
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
    w = max(24, max(len(r["model"]) for r in rows)) if rows else 24
    print(f"{'dataset':9s} {'model':{w}s} {'best size':22s} {'seeds':>5s}  "
          f"{'yours':>14s}  {'paper':>14s}  {'diff':>6s}  verdict")
    for r in rows:
        yours = f"{r['your_acc']:.2f} +/- {r['your_acc_sd']:.2f}"
        paper = f"{r['paper_acc']:.2f} +/- {r['paper_acc_sd']:.2f}"
        print(f"{r['dataset']:9s} {r['model']:{w}s} {r['best_size']:22s} {r['seeds']:>5d}  "
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
    p.add_argument("--select", default="default", choices=["default", "best-val"],
                   help="which variant of each model the comparison figures (7, 8) and control "
                        "tables use when constants were tuned: the default constants, or the "
                        "variant with the best VALIDATION accuracy")
    args = p.parse_args()

    runs = load_runs(args.results, args.dataset)
    if not runs:
        raise SystemExit(f"No results found in {args.results}/. Run base_model.py first.")
    configs = group_runs(runs)
    # Models whose apical dendrites read a coarse view are a separate experiment: they get
    # their own table and figure and stay out of the comparisons with the paper.
    context_all = [c for c in configs if paper_comparable(c)]
    comparable = [c for c in context_all if not has_context(c)]
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

    # Comparison figures and tables see one variant per model (constants chosen by --select)
    selected = select_variants(comparable, args.select)
    print(f"Variant selection for figures 7-8 and the control tables: {args.select}")

    fig_paper_comparison(best, datasets, out)
    fig_size_sweep(comparable, datasets, out)
    fig_dendrites_per_soma(comparable, datasets, out)
    fig_learning_curves(best, out)
    fig_control_homogeneous(selected, datasets, out)
    fig_ratio_sweep(selected, datasets, out)
    fig_constants_sweep(comparable, datasets, out)
    fig_size_generalisation(comparable, selected, datasets, out)
    print_summary(write_summary(best, n_configs, out))
    write_control_homogeneous(selected, out)
    write_ratio_sweep(selected, out)
    write_constants_sweep(comparable, out)
    write_paired(comparable, selected, out)
    fig_data_efficiency(configs, datasets, out)
    write_apical_context(context_all, datasets, out)


if __name__ == "__main__":
    main()
