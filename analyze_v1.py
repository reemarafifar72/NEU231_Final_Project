#!/usr/bin/env python3
"""
Inspect trained dANNs that use the V1 dendritic nonlinearities (`--nonlin v1`, or the
single-channel controls `supra`, `sub`, `ca`): which of the nonlinearities does each soma
actually use, and do the dendrites ever reach the calcium-spike threshold?

    python analyze_v1.py results/dann_v1_local_fmnist_L1_d16_s256_seed1 [more tags ...]
                         [--data-dir data] [--out figures]

For every tag (a results path without .json/.pt) it prints, per layer, the fraction of each
soma's absolute cable weight that reads each channel (only meaningful for v1, which has
three channels per dendrite), and how many somata are dominated by each channel. With
--data-dir it also pushes one validation batch through the first dendritic layer and reports
the scale of the dendritic pre-activations u, the fraction of them above the Ca threshold, and
each channel's share of the actual drive to the somata (mean |w * f_k(u)| over the batch),
which accounts for the channels' different output scales.

Figures (for the first tag):
    5_v1_nonlinearities.png   the f_k(u) curves of that run's channels, with its constants
    6_v1_channel_usage.png    per-soma stacked bars of the cable-weight fractions (v1 only)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base_model import DendriticANN, V1DendriticNonlinearity, load_data
from plot_results import COLOR, GRID, INK_2, MUTED, SURFACE, _finish, plt

IMG = {"fmnist": ((28, 28), 1), "cifar10": ((32, 32), 3)}
CHANNEL_COLOR = {"supra": COLOR["global"], "sub": COLOR["local"], "ca": COLOR["random"]}
NONLIN_KEYS = ("supra_p", "sub_sat", "ca_thresh", "ca_gain", "ca_amp")
BANK_NONLINS = ("v1", "supra", "sub", "ca")


def load(tag):
    """Rebuild the model from its results JSON and load the saved weights (CPU)."""
    tag = Path(str(tag).removesuffix(".json").removesuffix(".pt"))
    r = json.loads(tag.with_suffix(".json").read_text())
    nonlin = r.get("nonlin", "leaky")
    if nonlin not in BANK_NONLINS:
        raise SystemExit(f"{tag.name} is a {nonlin} run; this script needs one of {BANK_NONLINS}")
    img_shape, channels = IMG[r["dataset"]]
    model = DendriticANN(img_shape, channels, 10, dends=r["dends"], soma=r["soma"],
                         synapses=r["synapses"], layers=r["layers"], sampling=r["sampling"],
                         seed=r["seed"], nonlin=nonlin,
                         nonlin_params={k: r[k] for k in NONLIN_KEYS})
    model.load_state_dict(torch.load(tag.with_suffix(".pt"), map_location="cpu"))
    model.eval()
    return model, r


@torch.no_grad()
def channel_weight_fraction(soma_layer, dends, soma, n_ch):
    """(n_ch, soma): share of each soma's absolute cable weight on each channel."""
    D = dends * soma
    W = (soma_layer.weight * soma_layer.mask).abs()  # (n_ch * D, soma), channel-major rows
    per_ch = torch.stack([W[k * D:(k + 1) * D].sum(0) for k in range(n_ch)])
    return per_ch / per_ch.sum(0).clamp_min(1e-12)


def report_fractions(frac, names, title):
    dominant = frac.argmax(0)
    print(f"  {title}")
    for k, name in enumerate(names):
        f = frac[k]
        print(f"    {name:5s}  weight share mean {f.mean():.3f}  (min {f.min():.3f}, "
              f"max {f.max():.3f})  dominant in {(dominant == k).sum().item():4d} somata")


@torch.no_grad()
def report_activity(model, r, data_dir, n=2048):
    """Push validation images through layer 0 and report what the channels actually do."""
    data, *_ = load_data(r["dataset"], data_dir, seed=r["seed"])
    x = torch.as_tensor(data["val"][0][:n])
    bank = model.dend_act
    names = bank.channels
    u = model.dend_layers[0](x)                          # (n, D)
    h = bank(u)                                          # (n, n_ch * D)
    D = u.shape[1]
    W = model.soma_layers[0].weight * model.soma_layers[0].mask  # (n_ch * D, soma)
    # Actual drive each channel delivers to the somata, averaged over images and somata
    drive = torch.stack([(h[:, k * D:(k + 1) * D] @ W[k * D:(k + 1) * D]).abs().mean()
                         for k in range(len(names))])
    share = drive / drive.sum().clamp_min(1e-12)
    above = (u > bank.ca_thresh).float()
    print(f"  activity on {n} validation images (layer 0):")
    print(f"    u: mean {u.mean():.3f}  sd {u.std():.3f}  max {u.max():.3f}  "
          f"frac(u > 0) {(u > 0).float().mean():.3f}")
    print(f"    Ca threshold {bank.ca_thresh}: frac of (image, dendrite) above {above.mean():.4f}; "
          f"frac of dendrites ever above {(above.max(0).values.mean()):.4f}")
    for k, name in enumerate(names):
        print(f"    {name:5s}  mean |output| {h[:, k * D:(k + 1) * D].abs().mean():.4f}  "
              f"share of somatic drive {share[k]:.3f}")
    if "ca" in names and above.mean() < 1e-3:
        print("    note: the Ca channel almost never fires; try a lower --ca-thresh")


def fig_curves(bank, path):
    u = torch.linspace(-2, 3, 500)[:, None]
    y = bank(u).numpy()
    u = u.squeeze(1).numpy()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, name in enumerate(bank.channels):
        ax.plot(u, y[:, k], color=CHANNEL_COLOR[name], label=name)
    if "ca" in bank.channels:
        ax.axvline(bank.ca_thresh, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"Ca threshold {bank.ca_thresh:g}", (bank.ca_thresh, ax.get_ylim()[1]),
                    xytext=(4, -2), textcoords="offset points", va="top", fontsize=8,
                    color=INK_2)
    ax.set_xlabel("dendritic pre-activation u")
    ax.set_ylabel("channel output f(u)")
    ax.set_title("V1 dendritic nonlinearities")
    ax.legend(loc="upper left")
    _finish(fig, path, f"{bank.extra_repr()}. The sub channel is ~u near 0, so it also "
                       "carries the linear term.")


def fig_usage(frac, names, path, label):
    frac = frac.numpy()
    order = np.argsort(-frac[0])  # sort somata by their first channel's share
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bottom = np.zeros(frac.shape[1])
    xs = np.arange(frac.shape[1])
    for k, name in enumerate(names):
        ax.bar(xs, frac[k, order], bottom=bottom, width=1.0, color=CHANNEL_COLOR[name],
               label=f"{name} (mean {frac[k].mean():.2f})", lw=0)
        bottom += frac[k, order]
    ax.set_xlim(-0.5, frac.shape[1] - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"soma (sorted by {names[0]} share)")
    ax.set_ylabel("share of |cable weight|")
    ax.set_title(f"Which nonlinearity each soma reads: {label}")
    ax.grid(False)
    ax.legend(loc="upper right", ncol=len(names), frameon=True, facecolor=SURFACE, edgecolor=GRID)
    _finish(fig, path, "Each bar is one soma; the segments are the fraction of its absolute "
                       "dendrite-to-soma weight on each channel (layer 0).")


def main():
    p = argparse.ArgumentParser(description="Inspect trained dANNs with V1 dendritic nonlinearities")
    p.add_argument("tags", nargs="+", help="results/<tag> (no extension)")
    p.add_argument("--data-dir", default=None, help="also measure channel activity on data")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, tag in enumerate(args.tags):
        model, r = load(tag)
        bank = model.dend_act
        print(f"{Path(tag).name}: test acc {100 * r['test_acc']:.2f}%  |  {bank}")
        if i == 0:
            fig_curves(bank, out / "5_v1_nonlinearities.png")
        if bank.n_channels == 1:
            print(f"  single-channel run ({bank.channels[0]}-only): every soma reads only that "
                  "channel, so there is no channel usage to report")
        else:
            for li, soma_layer in enumerate(model.soma_layers):
                frac = channel_weight_fraction(soma_layer, r["dends"], r["soma"], bank.n_channels)
                report_fractions(frac, bank.channels,
                                 f"layer {li}: cable-weight share per channel")
                if i == 0 and li == 0:
                    fig_usage(frac, bank.channels, out / "6_v1_channel_usage.png",
                              f"{r['dends']} dends x {r['soma']} somata, seed {r['seed']}")
        if args.data_dir:
            report_activity(model, r, args.data_dir)


if __name__ == "__main__":
    main()
