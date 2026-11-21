#!/usr/bin/env python3
"""
Inspect trained dANNs that use the V1 dendritic nonlinearities (`--nonlin v1`, the
single-channel controls `supra`, `sub`, `ca`, or the per-dendrite ratio mode `branch`):
which of the nonlinearities does each soma actually use, and do the dendrites ever reach
the calcium-spike threshold?

    python analyze_v1.py results/dann_v1_local_fmnist_L1_d16_s256_seed1 [more tags ...]
                         [--data-dir data] [--out figures]

For every tag (a results path without .json/.pt) it prints, per layer, the fraction of each
soma's absolute cable weight that reads each nonlinearity, and how many somata are dominated
by each. For v1 the channels are the three outputs of every dendrite; for branch they are
the dendrites of each type (so this shows whether a soma downweights, say, its Ca dendrite).
With --data-dir it also pushes one validation batch through the first dendritic layer and
reports the scale of the dendritic pre-activations u, the fraction above the Ca threshold,
and each nonlinearity's share of the actual drive to the somata (mean |w * f(u)| over the
batch), which accounts for the channels' different output scales.

For `--nonlin gate` runs it reports instead how the apical Ca plateau gates the basal drive:
apical vs basal weight share, how often the gate is open, the mean amplification, and the
correlation between basal drive and gate (coincidence).

Figures (for the first tag):
    5_v1_nonlinearities.png   the f_k(u) curves of that run's channels, with its constants
    6_v1_channel_usage_by_{supra,sub,ca}.png
                              per-soma stacked bars of the cable-weight fractions (v1 and
                              branch runs), somata sorted by that channel's share
    10_gate_activity.png      gate runs: histogram of gate values, basal drive vs gate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base_model import CONSTANT_DEFAULTS, DendriticANN, load_data
from plot_results import COLOR, GRID, INK_2, MUTED, NONLIN_COLOR, SURFACE, _finish, plt

IMG = {"fmnist": ((28, 28), 1), "cifar10": ((32, 32), 3)}
CHANNEL_COLOR = {"supra": COLOR["global"], "sub": COLOR["local"], "ca": COLOR["random"]}
BANK_NONLINS = ("v1", "supra", "sub", "ca", "branch", "gate")


def load(tag, require_bank=True):
    """Rebuild the model from its results JSON and load the saved weights (CPU).

    require_bank=False also accepts the LeakyReLU baseline (robustness.py evaluates it).
    """
    tag = Path(str(tag).removesuffix(".json").removesuffix(".pt"))
    # not with_suffix: tags contain dots (..._sat0.2_seed1), which it would cut at
    r = json.loads(Path(f"{tag}.json").read_text())
    nonlin = r.get("nonlin", "leaky")
    if require_bank and nonlin not in BANK_NONLINS:
        raise SystemExit(f"{tag.name} is a {nonlin} run; this script needs one of {BANK_NONLINS}")
    img_shape, channels = IMG[r["dataset"]]
    consts = {k: (r.get(k) if r.get(k) is not None else d) for k, d in CONSTANT_DEFAULTS.items()}
    apical_frac = consts.pop("apical_frac")
    model = DendriticANN(img_shape, channels, 10, dends=r["dends"], soma=r["soma"],
                         synapses=r["synapses"], layers=r["layers"], sampling=r["sampling"],
                         seed=r["seed"], nonlin=nonlin, mix=r.get("mix", "1:1:1"),
                         nonlin_params=consts, basal_nonlin=r.get("basal_nonlin") or "leaky",
                         apical_frac=apical_frac, apical_input=r.get("apical_input") or "same",
                         apical_pool=r.get("apical_pool") or 4)
    model.load_state_dict(torch.load(f"{tag}.pt", map_location="cpu"))
    model.eval()
    return model, r


def layer0_input(model, x):
    """What the first dendritic layer reads: the image, plus the pooled view if apical is coarse."""
    return model.with_coarse_view(x) if model.coarse is not None else x


@torch.no_grad()
def gate_ablation(model, data, batch=2000):
    """Test accuracy with the gate intact, shuffled across images, and fixed at its mean.

    Shuffling keeps every soma's distribution of gate values but breaks the link to the image,
    so the drop from intact to shuffled is the image-specific information the gate carries.
    Single-layer models only.
    """
    g0 = model.soma_layers[0]
    x, y = torch.as_tensor(data["test"][0]), torch.as_tensor(data["test"][1])
    basal, gate = [], []
    for i in range(0, len(x), batch):
        h = model.dend_act(model.dend_layers[0](layer0_input(model, x[i:i + batch])))
        basal.append(g0.basal(h))
        g0(h)
        gate.append(g0.last_gate)
    basal, gate = torch.cat(basal), torch.cat(gate)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))

    def acc(gt):
        logits = model.output(model.soma_act(basal * (1 + g0.ca_amp * gt)))
        return 100 * (logits.argmax(1) == y).float().mean().item()

    return dict(intact=acc(gate), shuffled=acc(gate[perm]),
                mean=acc(gate.mean(0, keepdim=True).expand_as(gate)), closed=acc(torch.zeros_like(gate)))


@torch.no_grad()
def report_gate(model, r, data_dir, out, n=2048, make_fig=True):
    """Ca-gate runs: how much the apical plateau actually gates the basal drive."""
    g0 = model.soma_layers[0]
    wb = (g0.basal.weight * g0.basal.mask).abs().sum(0)
    wa = (g0.apical.weight * g0.apical.mask).abs().sum(0)
    share = wa / (wa + wb).clamp_min(1e-12)
    print(f"  gate model: {g0.n_basal} basal + {g0.n_apical} apical dendrites per soma; apical "
          f"share of |cable weight| mean {share.mean():.3f} (min {share.min():.3f}, max {share.max():.3f})"
          f" vs {g0.n_apical / (g0.n_basal + g0.n_apical):.3f} by count")
    if not data_dir:
        print("  (pass --data-dir to measure gate activity on validation images)")
        return
    data, *_ = load_data(r["dataset"], data_dir, seed=r["seed"])
    x = torch.as_tensor(data["val"][0][:n])
    h = model.dend_act(model.dend_layers[0](layer0_input(model, x)))
    basal = g0.basal(h)
    g0(h)  # sets last_gate
    gate = g0.last_gate                                   # (n, soma) in [0, 1]
    corr = torch.corrcoef(torch.stack([basal.flatten(), gate.flatten()]))[0, 1]
    print(f"  gate activity on {n} validation images (layer 0):")
    print(f"    Ca gate: mean {gate.mean():.3f}; frac of (image, soma) above 0.5 {(gate > 0.5).float().mean():.3f}; "
          f"frac of somata with mean gate > 0.5 {(gate.mean(0) > 0.5).float().mean():.3f}")
    print(f"    amplification (1 + g*Ca): mean {(1 + g0.ca_amp * gate).mean():.3f}; "
          f"corr(basal drive, gate) {corr:.3f}")
    if len(model.soma_layers) == 1:
        ab = gate_ablation(model, data)
        print(f"    gate ablation, test acc: intact {ab['intact']:.2f}%  |  gate shuffled across "
              f"images {ab['shuffled']:.2f}%  |  gate fixed at each soma's mean {ab['mean']:.2f}%"
              f"  |  gate closed {ab['closed']:.2f}%")
        print(f"    image-specific contribution of the gate (intact - shuffled): "
              f"{ab['intact'] - ab['shuffled']:+.2f} pp")
    if (gate > 0.5).float().mean() < 1e-3:
        print("    note: the gate almost never opens; try a lower --ca-thresh")
    if (gate > 0.5).float().mean() > 0.999:
        print("    note: the gate is always open, so it acts as a constant gain; try a higher --ca-thresh")
    if make_fig:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].hist(gate.flatten().numpy(), bins=50, color=NONLIN_COLOR["gate"], lw=0)
        axes[0].set_xlabel("Ca gate value (per image, per soma)")
        axes[0].set_ylabel("count")
        axes[0].set_title("How open the apical gate is")
        idx = torch.randperm(basal.numel(), generator=torch.Generator().manual_seed(0))[:4000]
        axes[1].scatter(basal.flatten()[idx].numpy(), gate.flatten()[idx].numpy(), s=5, alpha=0.35,
                        color=NONLIN_COLOR["gate"], lw=0)
        axes[1].set_xlabel("basal drive B")
        axes[1].set_ylabel("Ca gate")
        axes[1].set_title(f"Coincidence: basal drive vs gate (r = {corr:.2f})")
        _finish(fig, out / "10_gate_activity.png",
                "Left: distribution of the apical Ca plateau over validation images and somata. "
                "Right: the gate only matters where basal drive is large; a positive correlation "
                "means apical and basal inputs tend to agree.")


def row_groups(act, D):
    """Which rows of the cable weight belong to each nonlinearity, and their names.

    v1:     channel-major blocks of D rows, one per channel.
    branch: the rows (dendrites) whose type is that channel.
    single: one group holding everything.
    """
    if hasattr(act, "type_idx"):
        return [(name, act.type_idx == k) for k, name in enumerate(act.channels)]
    masks = []
    for k, name in enumerate(act.channels):
        m = torch.zeros(act.n_channels * D, dtype=torch.bool)
        m[k * D:(k + 1) * D] = True
        masks.append((name, m))
    return masks


@torch.no_grad()
def channel_weight_fraction(soma_layer, groups):
    """(n_groups, soma): share of each soma's absolute cable weight on each nonlinearity."""
    W = (soma_layer.weight * soma_layer.mask).abs()
    per = torch.stack([W[m].sum(0) for _, m in groups])
    return per / per.sum(0).clamp_min(1e-12)


def report_fractions(frac, names, title):
    dominant = frac.argmax(0)
    print(f"  {title}")
    for k, name in enumerate(names):
        f = frac[k]
        print(f"    {name:5s}  weight share mean {f.mean():.3f}  (min {f.min():.3f}, "
              f"max {f.max():.3f})  dominant in {(dominant == k).sum().item():4d} somata")


@torch.no_grad()
def report_activity(model, r, data_dir, n=2048):
    """Push validation images through layer 0 and report what the nonlinearities actually do."""
    data, *_ = load_data(r["dataset"], data_dir, seed=r["seed"])
    x = torch.as_tensor(data["val"][0][:n])
    act = model.dend_act
    u = model.dend_layers[0](layer0_input(model, x))  # (n, D)
    h = act(u)                                           # (n, n_ch * D)
    W = model.soma_layers[0].weight * model.soma_layers[0].mask  # (n_ch * D, soma)
    groups = row_groups(act, u.shape[1])
    # Actual drive each nonlinearity delivers to the somata, averaged over images and somata
    drive = torch.stack([(h[:, m] @ W[m]).abs().mean() for _, m in groups])
    share = drive / drive.sum().clamp_min(1e-12)
    above = (u > act.ca_thresh).float()
    print(f"  activity on {n} validation images (layer 0):")
    print(f"    u: mean {u.mean():.3f}  sd {u.std():.3f}  max {u.max():.3f}  "
          f"frac(u > 0) {(u > 0).float().mean():.3f}")
    print(f"    Ca threshold {act.ca_thresh}: frac of (image, dendrite) above {above.mean():.4f}; "
          f"frac of dendrites ever above {(above.max(0).values.mean()):.4f}")
    for k, (name, m) in enumerate(groups):
        if not m.any():
            print(f"    {name:5s}  no dendrites of this type")
            continue
        print(f"    {name:5s}  mean |output| {h[:, m].abs().mean():.4f}  "
              f"share of somatic drive {share[k]:.3f}")
    if "ca" in [g[0] for g in groups] and above.mean() < 1e-3:
        print("    note: the Ca channel almost never fires; try a lower --ca-thresh")


@torch.no_grad()
def report_dendrite_states(model, r, data_dir, out, n=2048, make_fig=True):
    """Are the Ca dendrites used as on/off switches? Looks at layer 0 on validation images.

    The Ca plateau is sigmoid(gain * (u - theta)) in [0, 1] (leak and amplitude left out, so
    0 is fully off and 1 fully on). Reports how much of the time a dendrite sits near either
    end, and whether each dendrite is selective (on for some images, off for others) or stuck.
    """
    act = model.dend_act
    bank = act.bank if hasattr(act, "type_idx") else act
    if "ca" not in bank.channels:
        return
    data, *_ = load_data(r["dataset"], data_dir, seed=r["seed"])
    x = torch.as_tensor(data["val"][0][:n])
    u = model.dend_layers[0](layer0_input(model, x))  # (n, D)
    if hasattr(act, "type_idx"):                                   # branch: only the Ca dendrites
        u = u[:, act.type_idx == list(act.channels).index("ca")]
    if u.shape[1] == 0:
        return
    ca = torch.sigmoid(bank.ca_gain * (u - bank.ca_thresh))        # (n, n_ca_dendrites)
    off, on = (ca < 0.1).float().mean().item(), (ca > 0.9).float().mean().item()
    rate = (ca > 0.5).float().mean(0)                              # per dendrite: frac of images on
    stuck_off, stuck_on = (rate < 0.02).float().mean().item(), (rate > 0.98).float().mean().item()
    print(f"  Ca dendrite states on {n} validation images (layer 0, {u.shape[1]:,} Ca dendrites, "
          f"threshold {bank.ca_thresh:g}, gain {bank.ca_gain:g}):")
    print(f"    (image, dendrite) pairs: off (<0.1) {off:.3f} | graded {1 - off - on:.3f} | "
          f"on (>0.9) {on:.3f}")
    print(f"    dendrites: always off {stuck_off:.3f} | selective {1 - stuck_off - stuck_on:.3f} "
          f"| always on {stuck_on:.3f}   (on = above 0.5 for an image; always = >98% of images)")
    print(f"    selective dendrites are on for a median {rate[(rate >= 0.02) & (rate <= 0.98)].median().item() if ((rate >= 0.02) & (rate <= 0.98)).any() else float('nan'):.3f} "
          f"of images; mean on-rate over all dendrites {rate.mean():.3f}")
    if make_fig:
        color = CHANNEL_COLOR["ca"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].hist(ca.flatten().numpy(), bins=50, color=color, lw=0)
        axes[0].set_xlabel("Ca output (per image, per dendrite)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Dendrite output: {100 * (off + on):.0f}% near 0 or 1")
        axes[1].hist(rate.numpy(), bins=50, range=(0, 1), color=color, lw=0)
        axes[1].set_xlabel("fraction of images for which the dendrite is on")
        axes[1].set_ylabel("dendrites")
        axes[1].set_title(f"Selectivity: {100 * (1 - stuck_off - stuck_on):.0f}% of dendrites switch")
        _finish(fig, out / "12_dendrite_states.png",
                f"{r['dataset']}, {r['nonlin']} dendrites, "
                f"{r['dends']} x {r['soma']}, seed {r['seed']}, Ca threshold {bank.ca_thresh:g}, "
                f"gain {bank.ca_gain:g}. Left: the Ca output sigmoid(gain (u - threshold)) over "
                "validation images and dendrites; mass at the two ends means dendrites act as "
                "on/off units. Right: how often each dendrite is on; dendrites at 0 or 1 never "
                "change state and carry no information about the image.")


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


def fig_usage(frac, names, path, label, sort_by, note):
    """Stacked bars of each soma's channel shares, somata sorted by one channel's share."""
    frac = frac.numpy()
    order = np.argsort(-frac[names.index(sort_by)])
    # the sorted channel goes at the bottom of the stack so its share reads as one clean edge
    stack = [sort_by] + [n for n in names if n != sort_by]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bottom = np.zeros(frac.shape[1])
    xs = np.arange(frac.shape[1])
    for name in stack:
        k = names.index(name)
        ax.bar(xs, frac[k, order], bottom=bottom, width=1.0, color=CHANNEL_COLOR[name],
               label=f"{name} (mean {frac[k].mean():.2f})", lw=0)
        bottom += frac[k, order]
    ax.set_xlim(-0.5, frac.shape[1] - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"soma (sorted by {sort_by} share, descending)")
    ax.set_ylabel("share of |cable weight|")
    ax.set_title(f"Which nonlinearity each soma reads: {label}")
    ax.grid(False)
    ax.legend(loc="upper right", ncol=len(names), frameon=True, facecolor=SURFACE, edgecolor=GRID)
    _finish(fig, path, note)


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
        act = model.dend_act
        branch = hasattr(act, "type_idx")
        bank = act.bank if branch else act
        print(f"{Path(tag).name}: test acc {100 * r['test_acc']:.2f}%  |  {act}")
        if model.nonlin == "gate":
            print(f"  {model.soma_layers[0]}")
            report_gate(model, r, args.data_dir, out, make_fig=(i == 0))
            if hasattr(bank, "channels") and i == 0:
                fig_curves(bank, out / "5_v1_nonlinearities.png")
            continue
        if i == 0:
            fig_curves(bank, out / "5_v1_nonlinearities.png")
        if not branch and act.n_channels == 1:
            print(f"  single-channel run ({act.channels[0]}-only): every soma reads only that "
                  "channel, so there is no channel usage to report"
                  + ("" if args.data_dir else " (pass --data-dir for dendrite activity)"))
        else:
            D = r["dends"] * r["soma"]
            groups = row_groups(act, D)
            names = [g[0] for g in groups]
            if branch:
                print(f"  branch run: {dict(zip(names, act.counts))} dendrites of each type per "
                      "soma; an unbiased soma would weight them in that proportion")
            for li, soma_layer in enumerate(model.soma_layers):
                frac = channel_weight_fraction(soma_layer, groups)
                report_fractions(frac, names, f"layer {li}: cable-weight share per nonlinearity")
                if i == 0 and li == 0:
                    label = f"{r['dends']} dends x {r['soma']} somata, seed {r['seed']}"
                    if branch:
                        label += f", mix {act.mix}"
                        note = ("Each bar is one soma; the segments are the fraction of its "
                                "absolute dendrite-to-soma weight on dendrites of each type "
                                f"(layer 0). Dendrites per soma: {dict(zip(names, act.counts))}.")
                    else:
                        note = ("Each bar is one soma; the segments are the fraction of its "
                                "absolute dendrite-to-soma weight on each channel (layer 0).")
                    for ch in names:  # one figure per sort order
                        fig_usage(frac, names, out / f"6_v1_channel_usage_by_{ch}.png", label,
                                  ch, note)
        if args.data_dir:
            report_activity(model, r, args.data_dir)
            report_dendrite_states(model, r, args.data_dir, out, make_fig=(i == 0))


if __name__ == "__main__":
    main()
