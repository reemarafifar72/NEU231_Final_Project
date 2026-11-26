#!/usr/bin/env python3
"""
Architecture sweep results: does a wider or deeper paper model (LeakyReLU dendrites,
dANN-LRF) beat the 1 layer x 16 dendrites x 256 somata base?

Reads the plain baseline runs (results/dann_local_<dataset>_L<layers>_d<dends>_s<soma>_seed<n>.json,
no other suffix) written by sweep_arch.sh and writes to --out (default figures/architecture/<dataset>):

    arch_summary.csv    one row per architecture: params, final and early-stopped test accuracy,
                        train accuracy, and the paired difference from the base (same seeds)
    16_architecture.png test accuracy vs trainable parameters, one line per depth, and the
                        train - test gap (overfitting) for the same models

    python arch_results.py --dataset fmnist
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results import INK, INK_2, MUTED, _finish, mean_sd, paired_stats

BASE = (1, 16, 256)
DEPTH_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#0f8b8d"}
PATTERN = r"dann_local_{ds}_L(\d+)_d(\d+)_s(\d+)_seed(\d+)\.json"


def load(results, dataset):
    pat = re.compile(PATTERN.format(ds=dataset) + "$")
    runs = defaultdict(dict)  # (layers, dends, soma) -> {seed: run}
    for p in Path(results).glob(f"dann_local_{dataset}_L*.json"):
        m = pat.fullmatch(p.name)
        if not m:
            continue  # other models, constants, weight decay, subsets, apical input
        r = json.loads(p.read_text())
        if r.get("nonlin", "leaky") != "leaky" or r.get("synapses", 16) != 16:
            continue
        L, d, s, seed = map(int, m.groups())
        runs[(L, d, s)][seed] = r
    return runs


def early_stop(by_seed):
    """Test accuracy at the best-validation epoch (older runs did not record it)."""
    return {k: r["test_acc_best_val"] for k, r in by_seed.items() if "test_acc_best_val" in r}


def summarise(runs):
    base = runs.get(BASE, {})
    base_acc = {k: r["test_acc"] for k, r in base.items()}
    base_es = early_stop(base)
    rows = []
    for (L, d, s), by_seed in sorted(runs.items()):
        rs = list(by_seed.values())
        acc = {k: r["test_acc"] for k, r in by_seed.items()}
        es = early_stop(by_seed)
        row = dict(arch=f"{L}x{d}x{s}", layers=L, dends=d, soma=s, params=rs[0]["n_params"],
                   n=len(rs), seeds=" ".join(map(str, sorted(by_seed))))
        row["test"], row["test_sd"] = mean_sd([100 * v for v in acc.values()])
        row["test_es"], row["test_es_sd"] = (mean_sd([100 * v for v in es.values()]) if es
                                             else (float("nan"), float("nan")))
        row["train"], _ = mean_sd([100 * r["train_acc"][-1] for r in rs])
        row["gap"] = row["train"] - row["test"]
        for key, a, b in (("vs_base", acc, base_acc), ("vs_base_es", es, base_es)):
            ps = paired_stats(a, b) if (L, d, s) != BASE else None
            if ps:
                row[key] = 100 * ps["mean"]
                row[key + "_sd"] = 100 * ps["sd"]
                row[key + "_p"] = ps["p"]
                row[key + "_wins"] = f"{ps['wins']}/{ps['n']}"
        rows.append(row)
    return rows


def best_by_validation(runs, min_seeds=3):
    """The architecture with the highest mean best-epoch validation accuracy (never test)."""
    scored = [(float(np.mean([max(r["val_acc"]) for r in by_seed.values()])), arch)
              for arch, by_seed in runs.items() if len(by_seed) >= min_seeds]
    return max(scored)[1] if scored else None


def write_csv(rows, path):
    cols = ["arch", "layers", "dends", "soma", "params", "n", "seeds", "test", "test_sd",
            "test_es", "test_es_sd", "train", "gap", "vs_base", "vs_base_sd", "vs_base_p",
            "vs_base_wins", "vs_base_es", "vs_base_es_sd", "vs_base_es_p", "vs_base_es_wins"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {path}")


def print_table(rows):
    print(f"\n{'arch (LxDxS)':>13} {'params':>8} {'n':>2}  {'test %':>13}  {'early-stop %':>13}"
          f"  {'train-test':>10}  {'paired vs 1x16x256':>22}")
    for r in sorted(rows, key=lambda r: -r["test"]):
        pv = (f"{r['vs_base']:+.2f} +/- {r['vs_base_sd']:.2f} ({r['vs_base_wins']})"
              if "vs_base" in r else ("base" if (r["layers"], r["dends"], r["soma"]) == BASE else ""))
        print(f"{r['arch']:>13} {r['params']:>8,} {r['n']:>2}  {r['test']:6.2f} +/- {r['test_sd']:.2f}"
              f"  {r['test_es']:6.2f} +/- {r['test_es_sd']:.2f}  {r['gap']:10.2f}  {pv:>22}")


def figure(rows, dataset, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for L in sorted({r["layers"] for r in rows}):
        rs = sorted((r for r in rows if r["layers"] == L), key=lambda r: r["params"])
        x = [r["params"] for r in rs]
        c = DEPTH_COLOR.get(L, INK)
        a1.errorbar(x, [r["test"] for r in rs], yerr=[r["test_sd"] for r in rs], fmt="o",
                    color=c, capsize=3, label=f"{L} layer{'s' if L > 1 else ''}")
        a2.plot(x, [r["gap"] for r in rs], "o", color=c)
        for r in rs:
            for ax, y in ((a1, r["test"]), (a2, r["gap"])):
                ax.annotate(f"{r['dends']}x{r['soma']}", (r["params"], y), xytext=(5, -3),
                            textcoords="offset points", fontsize=7, color=INK_2)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
    base = next((r for r in rows if (r["layers"], r["dends"], r["soma"]) == BASE), None)
    if base:
        a1.axhline(base["test"], color=MUTED, lw=1, ls="--", zorder=0)
        a1.annotate("base 1x16x256", (0, base["test"]), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", fontsize=8, color=MUTED)
    a1.set_ylabel("test accuracy (%), final epoch")
    a1.set_title(f"{dataset}: accuracy vs size and depth")
    a1.legend(loc="lower right")
    a2.set_ylabel("train - test accuracy (pp)")
    a2.set_title("overfitting")
    _finish(fig, path, note="Points: mean +/- SD over seeds; labels are dendrites x somata per "
                            "layer. LeakyReLU dendrites, local receptive fields, 16 synapses.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None)
    p.add_argument("--print-best", action="store_true",
                   help="print the best architecture by validation accuracy as 'LAYERS DENDS SOMA' and exit")
    a = p.parse_args()
    if a.print_best:
        best = best_by_validation(load(a.results, a.dataset))
        if best is None:
            raise SystemExit("no architecture with 3 seeds")
        print(*best)
        return
    out = Path(a.out or f"figures/architecture/{a.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    rows = summarise(load(a.results, a.dataset))
    if not rows:
        raise SystemExit("no matching runs")
    print_table(rows)
    write_csv(rows, out / "arch_summary.csv")
    figure(rows, a.dataset, out / "16_architecture.png")


if __name__ == "__main__":
    main()
