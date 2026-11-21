#!/usr/bin/env python3
"""
How well do trained dANNs hold up when the test images are corrupted?

    python robustness.py --dataset fmnist --out figures/control1
    python robustness.py --dataset cifar10 --results $OUT --out figures/cifar10   # Colab, CPU is fine

No training: every run in --results that has saved weights (.pt), uses the paper's settings
and matches --size is evaluated on the test set under two corruptions, each at several levels:

    noise      Gaussian pixel noise with the given SD, clipped back to [0, 1]
    occlusion  one grey square covering the given fraction of the image, at a random position

The corruption is generated once per level from a fixed seed and is identical for every
model, so differences between models are not noise in the corruption. Level 0 reproduces
each run's saved test accuracy. Models are compared with the LeakyReLU baseline trained with
the same seed (same data split and wiring).

Each run's numbers are saved to <results>/robustness_<dataset>.cache as it finishes, so an
interrupted call can simply be run again and only new runs are evaluated. On a CPU allow up to
a minute per CIFAR-10 run (9 forward passes over the test set); a CUDA GPU is used if present.

Writes robustness.csv (one row per run and level), robustness_paired.csv (each model minus
the baseline at each level) and 13_robustness.png.
"""
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_v1 import IMG, load
from base_model import evaluate, load_data
from plot_results import (DATASET_NAME, MUTED, SURFACE, _finish, mean_sd,
                          nonlin_color, nonlin_label, nonlin_rank, nonlin_style,
                          paired_stats, paper_comparable, plt, variant_of)

NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
OCCLUSION_LEVELS = (0.0, 0.1, 0.25, 0.4)
PERTURBATIONS = (("noise", NOISE_LEVELS, "Gaussian noise SD (pixel range 0 to 1)"),
                 ("occlusion", OCCLUSION_LEVELS, "Fraction of the image covered by a grey square"))
# Shown in the figure when present (the CSVs list every model): baseline, supra, the steep bounded
# dendrites, the mixed model, and the apical-context models.
FIGURE_VARIANTS = ("leaky", "supra", "ca@thr0", "sub@sat0.2", "v1@thr0", "gate", "leaky@apcoarse",
                   "gate@apcoarse")


def perturb(x, kind, level, img_shape, channels, seed=0):
    """Corrupted copy of flat test images x (N, pixels) in [0, 1]. Deterministic in `seed`."""
    if level == 0:
        return x
    gen = torch.Generator().manual_seed(seed)
    if kind == "noise":
        return (x + level * torch.randn(x.shape, generator=gen)).clamp_(0, 1)
    M, N = img_shape
    side = max(1, int(round(math.sqrt(level * M * N))))
    img = x.view(len(x), M, N, channels).clone()  # channels-last, as load_data flattens
    rows = torch.randint(0, M - side + 1, (len(x),), generator=gen)
    cols = torch.randint(0, N - side + 1, (len(x),), generator=gen)
    for i in range(len(x)):
        img[i, rows[i]:rows[i] + side, cols[i]:cols[i] + side, :] = 0.5
    return img.view(len(x), -1)


CACHE_VERSION = 1  # bump if perturb() or the levels' meaning changes


def _cache_key(kind, level):
    return f"{kind}:{level:g}"


def _load_cache(path):
    try:
        cache = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return cache if cache.get("version") == CACHE_VERSION else {}


def _save_cache(path, cache):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(cache, version=CACHE_VERSION)))
    tmp.replace(path)


@torch.no_grad()
def evaluate_runs(results, dataset, size, data_dir):
    """[{variant, seed, kind, level, acc}] for every usable run, plus the tags skipped.

    Each run's accuracies are saved to <results>/robustness_<dataset>.cache as soon as it is
    evaluated, so an interrupted call loses nothing and a later call only evaluates new runs.
    """
    dends, soma = size
    img_shape, channels = IMG[dataset]
    cache_path = Path(results) / f"robustness_{dataset}.cache"
    cache = _load_cache(cache_path)
    todo, skipped = [], []
    for path in sorted(Path(results).glob("*.json")):
        runs = load_runs_one(path, dataset)
        if not runs:
            continue
        r = runs[0]
        if (r["dends"], r["soma"]) != (dends, soma) or not paper_comparable(r):
            continue
        if path.stem not in cache and not path.with_suffix(".pt").exists():
            skipped.append(path.stem)
            continue
        todo.append((path, r))
    needed = [(kind, level) for kind, levels, _ in PERTURBATIONS for level in levels]
    new = [p.stem for p, _ in todo
           if any(_cache_key(k, lv) not in cache.get(p.stem, {}) for k, lv in needed)]
    print(f"  {len(todo)} runs, {len(todo) - len(new)} already evaluated (cached), {len(new)} to "
          f"evaluate" + ("; on a CPU allow up to a minute per run" if new and dataset == "cifar10"
                           and not torch.cuda.is_available() else ""))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, test, corrupted, done, t0 = [], None, {}, 0, time.time()
    for path, r in todo:
        accs = cache.setdefault(path.stem, {})
        if path.stem in new:
            model = load(path, require_bank=False)[0].to(dev)
            if test is None:  # the test set does not depend on the seed
                data, *_ = load_data(dataset, data_dir, seed=r["seed"])
                test = (torch.as_tensor(data["test"][0]), torch.as_tensor(data["test"][1]).to(dev))
            clean = None
            for kind, level in needed:
                if _cache_key(kind, level) in accs:
                    continue
                if level == 0 and clean is not None:  # level 0 is the clean test set for both
                    accs[_cache_key(kind, level)] = clean
                    continue
                if (kind, level) not in corrupted:
                    # generated on the CPU so the corruption is the same on every device
                    corrupted[(kind, level)] = perturb(test[0], kind, level, img_shape,
                                                       channels).to(dev)
                _, acc = evaluate(model, corrupted[(kind, level)], test[1], 2000)
                accs[_cache_key(kind, level)] = 100 * acc
                if level == 0:
                    clean = 100 * acc
            _save_cache(cache_path, cache)
            done += 1
            left = (time.time() - t0) / done * (len(new) - done)
            print(f"  evaluated {path.stem}  ({done}/{len(new)}, about {left / 60:.0f} min left)",
                  flush=True)
        for kind, level in needed:
            records.append(dict(variant=variant_of(r), seed=r["seed"], kind=kind, level=level,
                                acc=accs[_cache_key(kind, level)], saved=100 * r["test_acc"]))
    return records, skipped


def load_runs_one(path, dataset):
    """load_runs for a single file (same validity checks), with defaults for old files."""
    try:
        r = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if "test_acc" not in r or "sampling" not in r or r.get("dataset") != dataset:
        return []
    r.setdefault("train_subset", None)
    return [r]


def paired_rows(records, dataset):
    """Each variant minus the baseline, per corruption level, on shared seeds."""
    acc = defaultdict(dict)  # (variant, kind, level) -> {seed: acc}
    for rec in records:
        acc[(rec["variant"], rec["kind"], rec["level"])][rec["seed"]] = rec["acc"]
    rows = []
    for (v, kind, level), by_seed in sorted(acc.items(), key=lambda kv: (nonlin_rank(kv[0][0]),
                                                                           kv[0][1], kv[0][2])):
        m, sd = mean_sd(list(by_seed.values()))
        st = paired_stats(by_seed, acc.get(("leaky", kind, level), {})) if v != "leaky" else None
        rows.append({
            "dataset": dataset, "variant": nonlin_label(v), "key": v, "perturbation": kind,
            "level": level, "seeds": len(by_seed), "acc": round(m, 3), "acc_sd": round(sd, 3),
            "paired_seeds": st["n"] if st else "", "diff": round(st["mean"], 3) if st else "",
            "diff_sd": round(st["sd"], 3) if st else "",
            "p": round(st["p"], 4) if st and math.isfinite(st["p"]) else "",
            "wins": f"{st['wins']}/{st['n']}" if st else "",
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def figure(rows, dataset, size, out, variants):
    by = defaultdict(dict)  # (key, kind) -> {level: row}
    for r in rows:
        by[(r["key"], r["perturbation"])][r["level"]] = r
    shown = [v for v in variants if any(k[0] == v for k in by)]
    fig, axes = plt.subplots(2, len(PERTURBATIONS), figsize=(5.4 * len(PERTURBATIONS), 6.6),
                             squeeze=False, sharex="col")
    for col, (kind, levels, xlabel) in enumerate(PERTURBATIONS):
        ax, axd = axes[0][col], axes[1][col]
        others = [v for v in shown if v != "leaky"]
        for v in shown:
            pts = by.get((v, kind), {})
            xs = [lv for lv in levels if lv in pts]
            color, style = nonlin_color(v), nonlin_style(v)
            if "@ap" in v:  # context models share a base colour with their local-input twin
                style = dict(style, ls=(0, (1, 2)), mfc=SURFACE)
            ax.errorbar(xs, [pts[x]["acc"] for x in xs], yerr=[pts[x]["acc_sd"] for x in xs],
                        color=color, mec=color if "@ap" in v else SURFACE, mew=1.2, capsize=3,
                        elinewidth=1.1, label=nonlin_label(v), **style)
            if v == "leaky":
                continue
            xd = [x for x in xs if pts[x]["diff"] != ""]
            if not xd:
                continue
            j = others.index(v)
            step = (levels[1] - levels[0]) * 0.12
            off = (j - (len(others) - 1) / 2) * step
            axd.errorbar([x + off for x in xd], [pts[x]["diff"] for x in xd],
                         yerr=[pts[x]["diff_sd"] for x in xd], color=color,
                         mec=color if "@ap" in v else SURFACE, mew=1.2, capsize=3,
                         elinewidth=1.1, **style)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{DATASET_NAME.get(dataset, dataset)}, {size[0]} x {size[1]}: {kind}")
        axd.axhline(0, color=MUTED, lw=1)
        axd.set_xlabel(xlabel)
        axd.set_ylabel("Accuracy minus baseline (pp)")
        axd.set_xticks(list(levels))
        if col == 0:
            ax.legend(loc="lower left", fontsize=8)
    _finish(fig, out / "13_robustness.png",
            "Top: test accuracy on corrupted test images, mean and SD over seeds (models were "
            "trained on clean images only). Bottom: each model minus the LeakyReLU baseline "
            "with the same seed, mean and SD of the paired differences; above zero means more "
            "robust than the baseline. The corruption is identical for every model.")


def main():
    p = argparse.ArgumentParser(description="Noise and occlusion robustness of trained dANNs")
    p.add_argument("--results", default="results", help="folder with the .json and .pt files")
    p.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    p.add_argument("--size", default="16x256", help="dendrites x somata of the models to evaluate")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="figures")
    p.add_argument("--variants", default=",".join(FIGURE_VARIANTS),
                   help="comma-separated variant keys drawn in the figure (CSVs list all)")
    args = p.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, skipped = evaluate_runs(args.results, args.dataset, size, args.data_dir)
    if skipped:
        print(f"  no saved weights (.pt), skipped {len(skipped)} runs: " + ", ".join(skipped[:6])
              + (" ..." if len(skipped) > 6 else ""))
    if not records:
        raise SystemExit("No runs with saved weights to evaluate.")
    bad = [r for r in records if r["level"] == 0 and abs(r["acc"] - r["saved"]) > 0.05]
    if bad:
        print(f"  WARNING: {len(bad)} clean evaluations differ from the saved test accuracy")
    write_csv(out / "robustness.csv",
              [dict(dataset=args.dataset, variant=nonlin_label(r["variant"]), key=r["variant"],
                    seed=r["seed"], perturbation=r["kind"], level=r["level"],
                    acc=round(r["acc"], 3)) for r in records])
    rows = paired_rows(records, args.dataset)
    write_csv(out / "robustness_paired.csv", rows)
    figure(rows, args.dataset, size, out, args.variants.split(","))

    wv = max(len(r["variant"]) for r in rows)
    print("\nRobustness: test accuracy (%) and paired difference from the baseline (pp)")
    for kind, levels, _ in PERTURBATIONS:
        print(f"{kind:10s} {'model':{wv}s}  " + "  ".join(f"{lv:>13g}" for lv in levels))
        for v in dict.fromkeys(r["key"] for r in rows):
            cells = {r["level"]: r for r in rows if r["key"] == v and r["perturbation"] == kind}
            label = nonlin_label(v)
            print(f"{'':10s} {label:{wv}s}  " + "  ".join(
                (f"{cells[lv]['acc']:5.1f}" + (f" ({cells[lv]['diff']:+5.1f})" if cells[lv]["diff"] != ""
                                                else "        ")) if lv in cells else " " * 13
                for lv in levels))
    print("  (in brackets: paired difference from the baseline; see robustness_paired.csv for "
          "SD, p and seeds won)")


if __name__ == "__main__":
    main()
