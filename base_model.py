#!/usr/bin/env python3
"""
Base model: dendritic ANN (dANN) from Chavlis & Poirazi (2025), following their
code at https://github.com/Poirazi-Lab/dendritic_anns.

Architecture (per dendro-somatic layer):
    input --(sparse synaptic weights)--> dendrites --LeakyReLU(0.1)-->
          --(structured cable weights: each dendrite -> one soma)--> somata --LeakyReLU(0.1)-->
    ... repeated `layers` times ... --> dense output layer --> softmax (inside the loss)

With --nonlin v1, the dendritic LeakyReLU is replaced by three fixed V1-style nonlinearities
applied in parallel to every dendrite (supralinear, sublinear, calcium spike). The soma then
learns a weighted sum over all (dendrite, channel) outputs. See V1DendriticNonlinearity.
--nonlin supra / sub / ca are the "homogeneous type" controls: every dendrite uses just that
one nonlinearity, with the baseline-sized cable layer. --nonlin branch --mix S:B:C gives each
dendrite ONE nonlinearity, with the ratio setting how many of a soma's dendrites are supra,
sub and Ca (12:3:1 at 16 dendrites = 12, 3 and 1); same cable layer as the baseline.
Results are tagged dann_<nonlin>_* (dann_branch12-3-1_* for ratios); --nonlin leaky (the
default) is the paper's model and keeps the original tag.

Matches the original code:
    - Masks built the same way (16 synapses per dendrite; random / local-RF / global-RF sampling).
    - Glorot-uniform init on the full dense weight shape, zero biases, then masked.
    - Adam (lr 1e-3), batch 128, cross-entropy, 90/10 train/validation split, pixels scaled to [0, 1].
    - Epochs: 25 for Fashion-MNIST, 50 for CIFAR-10 (the values in their main.py).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets



########## Connectivity masks ##########

def neighborhood(shape, center, radius=1, perimeter=False):
    """
    Adapted from nb_vals in receptive_fields.py.

    Extract the indices of neighboring elements around a specific center in a matrix.
    (row, col) pixels within Chebyshev distance `radius` of `center`, clipped to the image.

    Returns the pixel coordinates around a center pixel.
    """
    M, N = shape
    r, c = int(center[0]), int(center[1])
    r0, r1 = max(0, r - radius), min(M - 1, r + radius)
    c0, c1 = max(0, c - radius), min(N - 1, c + radius)
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    if perimeter:
        # Chebyshev distance formula: max(|x1 - x2|, |y1 - y2|)
        # We only keep coordinates where the distance is exactly `size` (the perimeter)
        keep = np.maximum(np.abs(rr - r), np.abs(cc - c)) == radius
        return np.column_stack((rr[keep], cc[keep]))
    return np.column_stack((rr.ravel(), cc.ravel()))


def grow_to(shape, center, n, rng):
    """
    Adapted from allocate_synapses in receptive_fields.py.

    Start from the 3x3 patch around `center`, add pixels from successive rings until there are n.

    For 16 synapses, that is the 9 central pixels plus 7 pixels picked at random from the surrounding ring.
    This produces a dendrite is receptive field: a small, local patch of the image.
    """
    idx = neighborhood(shape, center, 1)
    radius = 2
    while len(idx) < n:
        ring = neighborhood(shape, center, radius, perimeter=True)
        if len(ring) == 0:
            break
        need = n - len(idx)
        if len(ring) > need:
            ring = ring[rng.choice(len(ring), size=need, replace=False)]
        idx = np.concatenate((idx, ring))
        radius += 1
    if len(idx) > n:
        idx = idx[rng.choice(len(idx), size=n, replace=False)]
    if len(idx) != n:
        raise ValueError(f"Could not find {n} pixels; image too small.")
    return idx


def dendrite_row(shape, center, synapses, channels, rng):
    """
    Adapted from allocate_synapses in receptive_fields.py.

    Builds the input mask for one dendrite. It calls _grow_to to choose 16 pixels near center.
    Then marks those pixels with 1 in an image-sized grid of zeros.
    """
    m = np.zeros(shape)
    idx = grow_to(shape, center, synapses, rng)
    m[idx[:, 0], idx[:, 1]] = 1
    if channels > 1:  # same pixels in every color channel
        m = np.repeat(m[:, :, None], channels, axis=2)
    return m.reshape(-1)


def synaptic_mask(shape, dends, soma, synapses, channels, sampling, rng):
    """
    Adapted from receptive_fields (line 315) + make_mask_matrix (line 226) + random_connectivity (line 73)

    Builds the full input -> dendrite mask, shape (784 inputs x 2,048 dendrites).
    Dendrites are grouped by soma: the first 16 belong to soma 1, the next 16 to soma 2, and so on
    """
    M, N = shape
    n_in = M * N * channels
    n_dend = dends * soma

    if sampling == "random":
        # dANN-R: synapses*dends*soma connections scattered at random over the whole matrix
        mask = np.zeros(n_in * n_dend, dtype=np.int8)
        mask[rng.choice(n_in * n_dend, synapses * n_dend, replace=False)] = 1
        return mask.reshape(n_in, n_dend)

    rows = np.zeros((n_dend, n_in), dtype=np.int8)
    if sampling == "local":
        # dANN-LRF: every dendrite gets its own random RF center
        flat = rng.choice(M * N, size=n_dend, replace=True)
        for k, center in enumerate(zip(*np.unravel_index(flat, (M, N)))):
            rows[k] = dendrite_row(shape, center, synapses, channels, rng)
    elif sampling == "global":
        # dANN-GRF: every soma gets a random center; its dendrites' centers lie around it.
        flat = rng.choice(M * N, size=soma, replace=True)
        k = 0
        for center in zip(*np.unravel_index(flat, (M, N))):
            for d_center in grow_to(shape, center, dends, rng):
                rows[k] = dendrite_row(shape, d_center, synapses, channels, rng)
                k += 1
    else:
        raise ValueError("sampling must be 'random', 'local' or 'global'")
    return rows.T


def cable_mask(dends, soma):
    """
    Adapted from connectivity in receptive_fields.py

    Builds the dendrite -> soma mask, shape (2,048 x 128).
    Dendrites 1-16 connect only to soma 1, dendrites 17-32 only to soma 2, and so on.
    """
    return np.kron(np.eye(soma, dtype=np.int8), np.ones((dends, 1), dtype=np.int8))


def grid_shape(n):
    """
    Adapted from make_masks in opt.py

    Lay out n somata as a near-square 2D grid (how the original feeds layer i into layer i+1)."""
    divisors = [j for j in range(1, n + 1) if n % j == 0]
    ix = len(divisors) // 2
    if len(divisors) % 2 == 0:
        return divisors[ix], divisors[ix - 1]
    return divisors[ix], divisors[ix]


########## Model ##########

class MaskedLinear(nn.Module):
    """Linear layer whose absent connections are held at zero by a fixed binary mask.

    Weights are stored as (in, out), as in Keras, so masks have the same orientation as
    the original code. Multiplying by the mask in the forward pass gives the masked
    entries zero gradient, which is equivalent to the original's gradient masking.
    """

    def __init__(self, n_in, n_out, mask=None):
        super().__init__()
        limit = math.sqrt(6.0 / (n_in + n_out))  # Keras glorot_uniform on the full dense shape
        self.weight = nn.Parameter(torch.empty(n_in, n_out).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.zeros(n_out))
        m = torch.ones(n_in, n_out) if mask is None else torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", m)

    def forward(self, x):
        return x @ (self.weight * self.mask) + self.bias

    def n_params(self):
        """Trainable parameters that actually exist (masked weights + biases)."""
        return int(self.mask.sum().item()) + self.bias.numel()


class V1DendriticNonlinearity(nn.Module):
    """Three fixed V1-style dendritic nonlinearities applied in parallel to every dendrite.

    Input  u: (B, D) dendritic pre-activations, D = dends * soma (the dendrite bias is included).
    Output:   (B, 3 * D), channel-major  [ supra(u) | sub(u) | ca(u) ],
              so column k * D + j is channel k of dendrite j.

        supra(u) = relu(u) ** p                   NMDA-spike-like supralinear boost
        sub(u)   = s * tanh(u / s)                passive saturation; ~u for |u| << s,
                                                  so it also carries the linear (leak) term
        ca(u)    = A * sigmoid(k * (u - theta))   high-threshold, all-or-none Ca2+ plateau

    With leak > 0 every channel also conducts passively: f_k(u) + leak * u. That is the slope
    LeakyReLU keeps for negative input, and it removes the dead (supra, u < 0) and saturated
    (sub, Ca) regimes of the fixed functions. leak = 0 reproduces the original channels.

    All constants are fixed hyperparameters (not learned) and this module has no parameters.
    Only the cable weights that read these channels (MaskedLinear(3 * D, soma)) are trained,
    so each soma learns how much of each nonlinearity to use from each of its dendrites.

    `channels` selects a subset, in CHANNELS order. With a single channel the output is (B, D):
    that is the "homogeneous type" control, where every dendrite uses one nonlinearity.
    """
    CHANNELS = ("supra", "sub", "ca")

    def __init__(self, supra_p=2.0, sub_sat=1.0, ca_thresh=1.0, ca_gain=10.0, ca_amp=1.0,
                 leak=0.0, channels=CHANNELS):
        super().__init__()
        if supra_p < 1.0:
            raise ValueError("supra_p must be >= 1: d/du relu(u)**p is unbounded at 0 for p < 1")
        if sub_sat <= 0 or ca_gain <= 0 or leak < 0:
            raise ValueError("sub_sat and ca_gain must be > 0 and leak >= 0")
        bad = [c for c in channels if c not in self.CHANNELS]
        if bad or not channels:
            raise ValueError(f"channels must be a non-empty subset of {self.CHANNELS}, got {channels}")
        self.channels = tuple(c for c in self.CHANNELS if c in channels)
        self.supra_p, self.sub_sat = float(supra_p), float(sub_sat)
        self.ca_thresh, self.ca_gain, self.ca_amp = float(ca_thresh), float(ca_gain), float(ca_amp)
        self.leak = float(leak)

    @property
    def n_channels(self):
        return len(self.channels)

    def _leak(self, u):
        return self.leak * u if self.leak else 0.0

    def supra(self, u):
        return F.relu(u).pow(self.supra_p) + self._leak(u)

    def sub(self, u):
        return self.sub_sat * torch.tanh(u / self.sub_sat) + self._leak(u)

    def ca(self, u):
        return self.ca_amp * torch.sigmoid(self.ca_gain * (u - self.ca_thresh)) + self._leak(u)

    def forward(self, u):
        outs = [getattr(self, c)(u) for c in self.channels]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    def extra_repr(self):
        return (f"channels={self.channels}, supra_p={self.supra_p}, sub_sat={self.sub_sat}, "
                f"ca_thresh={self.ca_thresh}, ca_gain={self.ca_gain}, ca_amp={self.ca_amp}, "
                f"leak={self.leak}")


def parse_mix(mix):
    """'12:3:1' -> (12.0, 3.0, 1.0): the supra:sub:ca dendrite ratio. Entries >= 0, sum > 0."""
    try:
        parts = tuple(float(p) for p in str(mix).split(":"))
    except ValueError:
        raise ValueError(f"mix must look like 'S:B:C', got {mix!r}") from None
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError(f"mix must be three non-negative numbers with a positive sum, got {mix!r}")
    return parts


def mix_counts(mix, dends):
    """Integer dendrite counts (n_supra, n_sub, n_ca) per soma that sum to `dends`.

    Largest-remainder rounding of ratio * dends: 12:3:1 at 16 -> (12, 3, 1);
    1:1:1 at 16 -> (6, 5, 5); 1:1:0 at 16 -> (8, 8, 0).
    """
    ratio = parse_mix(mix)
    exact = [r / sum(ratio) * dends for r in ratio]
    counts = [int(math.floor(e)) for e in exact]
    for k in sorted(range(3), key=lambda k: exact[k] - counts[k], reverse=True)[:dends - sum(counts)]:
        counts[k] += 1
    assert sum(counts) == dends
    return tuple(counts)


class BranchTypeNonlinearity(nn.Module):
    """One V1 nonlinearity per dendrite ("mixed branches").

    The mix ratio supra:sub:ca sets how many of each soma's `dends` dendrites use each
    nonlinearity; within a soma the first n_supra dendrites are supralinear, the next n_sub
    sublinear and the rest calcium-spike. Dendrite order is arbitrary (receptive-field centers
    are random), so the layout is fixed. Input and output are (B, D), so the cable layer keeps
    the baseline shape and parameter count. No parameters, no RNG.
    """

    def __init__(self, dends, soma, mix="1:1:1", **constants):
        super().__init__()
        self.bank = V1DendriticNonlinearity(**constants)  # all three channels, same constants
        self.mix, self.counts = str(mix), mix_counts(mix, dends)
        for name, ratio, n in zip(self.bank.CHANNELS, parse_mix(mix), self.counts):
            if ratio > 0 and n == 0:
                print(f"warning: mix {mix} gives 0 {name} dendrites per soma at {dends} dendrites")
        per_soma = sum(([k] * n for k, n in enumerate(self.counts)), [])
        type_idx = torch.tensor(per_soma * soma, dtype=torch.long)  # (D,) channel index per dendrite
        self.register_buffer("type_idx", type_idx)
        self.register_buffer("onehot", F.one_hot(type_idx, 3).T.float())  # (3, D)

    @property
    def channels(self):
        return self.bank.channels

    @property
    def n_channels(self):  # channels per dendrite reaching the soma
        return 1

    @property
    def ca_thresh(self):
        return self.bank.ca_thresh

    def forward(self, u):
        h = self.bank(u).view(u.shape[0], 3, -1)  # (B, 3, D), channel-major
        return (h * self.onehot).sum(1)           # (B, D): each dendrite's own channel

    def extra_repr(self):
        return f"mix={self.mix} -> counts {self.counts} per soma; {self.bank.extra_repr()}"


def apical_flags(dends, soma, apical_frac):
    """Which dendrites are apical: the last round(apical_frac * dends) of every soma's block.

    Returns a boolean array of length dends * soma. Shared by the Ca gate and --apical-input so
    the two always mean the same dendrites.
    """
    n_apical = max(1, int(round(apical_frac * dends)))
    if n_apical >= dends:
        raise ValueError(f"apical_frac {apical_frac} leaves no basal dendrites at {dends} dendrites")
    return np.array(([False] * (dends - n_apical) + [True] * n_apical) * soma)


class CaGateSoma(nn.Module):
    """Soma whose basal drive is multiplied by an apical Ca2+ plateau (BAC-firing-like).

    Each soma's `dends` dendrites are split into basal (the first n_basal) and apical (the
    rest). Two masked cable layers read them separately:

        B = sum_basal  w * h + b          basal drive            (B, soma)
        A = sum_apical v * h + c          apical drive           (B, soma)
        Ca = sigmoid(k * (A - theta))     apical plateau in [0, 1]
        out = B * (1 + g * Ca)            apical gates basal; soma_act is applied afterwards

    Apical input alone does nothing (B = 0), basal alone gives the normal response, and both
    together amplify it by up to (1 + g): the coincidence detection a weighted sum of fixed
    functions cannot express. Note that a negative B is amplified too; the soma's LeakyReLU
    then clips it to slope 0.1. Masked weights total D, as in the baseline; there are two bias
    vectors instead of one. theta applies to the learned sum A, so the apical bias can shift
    the effective threshold.
    """

    def __init__(self, dends, soma, apical_frac=0.25, ca_thresh=1.0, ca_gain=10.0, ca_amp=1.0):
        super().__init__()
        is_apical = apical_flags(dends, soma, apical_frac)  # (D,)
        n_apical = int(is_apical[:dends].sum())
        self.n_basal, self.n_apical = dends - n_apical, n_apical
        self.ca_thresh, self.ca_gain, self.ca_amp = float(ca_thresh), float(ca_gain), float(ca_amp)
        cm = cable_mask(dends, soma)
        self.basal = MaskedLinear(dends * soma, soma, cm * (~is_apical)[:, None])
        self.apical = MaskedLinear(dends * soma, soma, cm * is_apical[:, None])
        self.register_buffer("is_apical", torch.as_tensor(is_apical))
        self.last_gate = None  # detached Ca values from the last forward pass, for analysis

    def forward(self, h):
        basal, apical = self.basal(h), self.apical(h)
        ca = torch.sigmoid(self.ca_gain * (apical - self.ca_thresh))
        self.last_gate = ca.detach()
        return basal * (1 + self.ca_amp * ca)

    def extra_repr(self):
        return (f"basal={self.n_basal}, apical={self.n_apical} dendrites per soma, "
                f"ca_thresh={self.ca_thresh}, ca_gain={self.ca_gain}, ca_amp={self.ca_amp}")


# Fixed constants of the biological nonlinearities and their defaults. Runs that change any of
# them get a tag suffix such as _thr0.5_leak0.1 so they never collide with default runs.
# plot_results.py rebuilds the same suffix to name the model variant: keep the two in sync.
CONSTANT_DEFAULTS = dict(supra_p=2.0, sub_sat=1.0, ca_thresh=1.0, ca_gain=10.0, ca_amp=1.0,
                         leak=0.0, apical_frac=0.25)
CONSTANT_SHORT = dict(supra_p="p", sub_sat="sat", ca_thresh="thr", ca_gain="gain", ca_amp="amp",
                      leak="leak", apical_frac="ap")


def constants_suffix(values, nonlin):
    """'_thr0.5_leak0.1' for the constants in `values` that differ from their defaults.

    Empty for the paper baseline and for default constants. apical_frac only counts for gate.
    """
    if nonlin == "leaky":
        return ""
    parts = []
    for key, default in CONSTANT_DEFAULTS.items():
        if key == "apical_frac" and nonlin != "gate":
            continue
        v = values.get(key, default)
        if v is not None and float(v) != default:
            parts.append(f"{CONSTANT_SHORT[key]}{float(v):g}")
    return "".join("_" + p for p in parts)


class DendriticANN(nn.Module):
    """
    Adapted from get_model in opt.py.

    Chavlis & Poirazi dANN: `layers` dendro-somatic layers + dense output.

    nonlin="leaky" (paper):
        784 pixels -> dendrites (2048) -> LeakyReLU -> somata (128) -> LeakyReLU -> output (10)
    nonlin="v1":
        784 pixels -> dendrites (2048) -> [supra | sub | Ca] (6144) -> somata (128) -> LeakyReLU
        -> output (10). The cable layer reads all three channels of each of its own dendrites.
    nonlin="supra" | "sub" | "ca" (homogeneous-type controls):
        as "leaky", but with that single V1 nonlinearity in place of the dendritic LeakyReLU.
        Same cable layer and parameter count as the baseline.
    nonlin="branch", mix="S:B:C" (per-dendrite ratio):
        each dendrite uses one V1 nonlinearity; the ratio sets how many of a soma's dendrites
        are supra, sub and Ca. Same cable layer and parameter count as the baseline.
    nonlin="gate" (apical Ca gate, see CaGateSoma):
        dendrites use `basal_nonlin` (LeakyReLU by default); each soma's basal drive is
        multiplied by (1 + g * Ca) where Ca is a plateau on its apical dendrites' drive.
        Same masked weights as the baseline plus one extra bias vector per layer.

    apical_input="coarse" (any nonlin): in the first layer, each soma's apical dendrites (the
        last apical_frac of its dendrites) take their synapses from an average-pooled copy of
        the image (apical_pool x apical_pool blocks) instead of from the image itself: a
        low-resolution, wide-field "context" view, as the apical tuft receives in cortex.
        Basal dendrites keep the wiring they have with apical_input="same". Pooling has no
        parameters and every dendrite keeps `synapses` inputs, so the parameter count is
        unchanged.
    """

    NONLINS = ("leaky", "v1", "supra", "sub", "ca", "branch", "gate")

    def __init__(self, img_shape, channels, num_classes, dends=16, soma=128,
                 synapses=16, layers=1, sampling="local", relu_slope=0.1, seed=0,
                 nonlin="leaky", nonlin_params=None, mix="1:1:1", basal_nonlin="leaky",
                 apical_frac=0.25, apical_input="same", apical_pool=4):
        super().__init__()
        if apical_input not in ("same", "coarse"):
            raise ValueError("apical_input must be 'same' or 'coarse'")
        if nonlin not in self.NONLINS:
            raise ValueError(f"nonlin must be one of {self.NONLINS}")
        if basal_nonlin not in ("leaky",) + V1DendriticNonlinearity.CHANNELS:
            raise ValueError("basal_nonlin must be leaky, supra, sub or ca")
        rng = np.random.default_rng(seed)
        self.nonlin, self.mix, self.basal_nonlin = nonlin, mix, basal_nonlin
        params = dict(nonlin_params or {})
        # the gate's soma takes the Ca constants; every other model keeps them in its dendrites
        gate_params = ({k: params.pop(k) for k in ("ca_thresh", "ca_gain", "ca_amp") if k in params}
                       if nonlin == "gate" else {})
        n_ch = 3 if nonlin == "v1" else 1  # channels per dendrite feeding the soma
        self.dend_layers = nn.ModuleList()
        self.soma_layers = nn.ModuleList()

        shape, ch = img_shape, channels
        n_in = shape[0] * shape[1] * ch
        self.coarse = None  # (rows, cols, channels, pool) of the image when apical input is coarse
        for layer in range(layers):
            syn = synaptic_mask(shape, dends, soma, synapses, ch, sampling, rng)
            if apical_input == "coarse" and layer == 0:
                # the fine mask above used the same rng draws as apical_input="same", so basal
                # wiring is identical for a given seed; the coarse mask has its own rng
                if shape[0] % apical_pool or shape[1] % apical_pool:
                    raise ValueError(f"apical_pool {apical_pool} does not divide image {shape}")
                c_shape = (shape[0] // apical_pool, shape[1] // apical_pool)
                c_syn = synaptic_mask(c_shape, dends, soma, synapses, ch, sampling,
                                      np.random.default_rng([seed, 1]))
                is_apical = apical_flags(dends, soma, apical_frac)
                syn = np.vstack([syn * ~is_apical[None, :], c_syn * is_apical[None, :]])
                n_in += c_shape[0] * c_shape[1] * ch
                self.coarse = (shape[0], shape[1], ch, apical_pool)
            self.dend_layers.append(MaskedLinear(n_in, dends * soma, syn))
            if nonlin == "gate":
                self.soma_layers.append(CaGateSoma(dends, soma, apical_frac, **gate_params))
            else:
                cm = cable_mask(dends, soma)
                if n_ch > 1:
                    # channel-major: rows [k*D, (k+1)*D) are channel k of every dendrite, with
                    # the same per-soma blocks, so each soma still reads only its own dendrites
                    cm = np.tile(cm, (n_ch, 1))
                self.soma_layers.append(MaskedLinear(n_ch * dends * soma, soma, cm))
            # next layer samples from this layer's somata arranged on a 2D grid
            shape, ch, n_in = grid_shape(soma), 1, soma

        self.output = MaskedLinear(soma, num_classes)  # fully connected
        # Dendritic and somatic nonlinearities (both LeakyReLU 0.1 in the paper)
        if nonlin == "gate":
            params.update(gate_params)  # the dendritic channel (if any) keeps the Ca constants
            self.dend_act = (nn.LeakyReLU(relu_slope) if basal_nonlin == "leaky"
                             else V1DendriticNonlinearity(channels=(basal_nonlin,), **params))
        elif nonlin == "leaky":
            self.dend_act = nn.LeakyReLU(relu_slope)
        elif nonlin == "v1":
            self.dend_act = V1DendriticNonlinearity(**params)
        elif nonlin == "branch":  # one nonlinearity per dendrite, mixed within each soma
            self.dend_act = BranchTypeNonlinearity(dends, soma, mix, **params)
        else:  # the same single nonlinearity for every dendrite
            self.dend_act = V1DendriticNonlinearity(channels=(nonlin,), **params)
        self.soma_act = nn.LeakyReLU(relu_slope)

    def with_coarse_view(self, x):
        """Append the average-pooled image (channels-last, like the input) to the flat input."""
        M, N, ch, pool = self.coarse
        img = x.view(len(x), M, N, ch).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(img, pool).permute(0, 2, 3, 1).reshape(len(x), -1)
        return torch.cat([x, pooled], dim=1)

    def forward(self, x):
        if self.coarse is not None:
            x = self.with_coarse_view(x)
        for dend, soma in zip(self.dend_layers, self.soma_layers):
            x = self.dend_act(dend(x))
            x = self.soma_act(soma(x))
        return self.output(x)  # logits; softmax is applied inside the cross-entropy loss

    def n_params(self):
        return sum(m.n_params() for m in self.modules() if isinstance(m, MaskedLinear))


########## Data ##########

def load_data(name, data_dir, val_split=0.1, seed=0, train_subset=None):
    """Return flattened float32 arrays in [0, 1] (channels-last, as in the original).

    train_subset: keep only the first N shuffled training images (after the validation split,
    which stays full). For quick checks only; such runs are tagged _nN and left out of plots.
    """
    if name == "fmnist":
        tr = datasets.FashionMNIST(data_dir, train=True, download=True)
        te = datasets.FashionMNIST(data_dir, train=False, download=True)
        xtr, ytr = tr.data.numpy(), tr.targets.numpy()
        xte, yte = te.data.numpy(), te.targets.numpy()
        img_shape, channels = xtr.shape[1:3], 1
    elif name == "cifar10":
        tr = datasets.CIFAR10(data_dir, train=True, download=True)
        te = datasets.CIFAR10(data_dir, train=False, download=True)
        xtr, ytr = tr.data, np.array(tr.targets)  # (N, 32, 32, 3)
        xte, yte = te.data, np.array(te.targets)
        img_shape, channels = xtr.shape[1:3], 3
    else:
        raise ValueError("dataset must be 'fmnist' or 'cifar10'")

    xtr = xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0
    xte = xte.reshape(len(xte), -1).astype(np.float32) / 255.0

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xtr))
    xtr, ytr = xtr[order], ytr[order]
    n_val = int(val_split * len(xtr))
    x_train, y_train = xtr[:-n_val], ytr[:-n_val]
    if train_subset:
        x_train, y_train = x_train[:train_subset], y_train[:train_subset]
    split = {
        "train": (x_train, y_train),
        "val": (xtr[-n_val:], ytr[-n_val:]),
        "test": (xte, yte),
    }
    return split, tuple(img_shape), channels, 10


########## Training ##########

def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, x, y, batch_size):
    """
    Runs the model on a dataset without training it and returns average loss and accuracy.
    Gradients are turned off here, which makes it faster.
    """
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        total_loss += F.cross_entropy(logits, y[i:i + batch_size], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    return total_loss / len(x), correct / len(x)


def train(model, data, epochs, lr, batch_size, device, seed, weight_decay=0.0):
    """The training loop. For each epoch it:

    1. Shuffles the training images.
    2. Goes through them in batches of 128, and for each batch, it:
        - predicts,
        - computes cross-entropy loss,
        - backpropagates,
        - takes an Adam step with learning rate 0.001.
    3. Records training loss and accuracy, then calls evaluate on the validation set and
    prints a progress line.

    4. After the last epoch it evaluates once on the test set. It returns a dictionary of all these numbers.

    Whenever validation accuracy reaches a new best, the test set is also evaluated and kept as
    test_acc_best_val / test_loss_best_val (early stopping chosen on validation, never on test).
    test_acc / test_loss stay the final-epoch values. weight_decay is Adam's L2 penalty on
    every parameter (0 = the paper's setting)."""

    to_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    xtr, ytr = to_t(data["train"][0], torch.float32), to_t(data["train"][1], torch.long)
    xva, yva = to_t(data["val"][0], torch.float32), to_t(data["val"][1], torch.long)
    xte, yte = to_t(data["test"][0], torch.float32), to_t(data["test"][1], torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr), generator=gen).to(device)
        run_loss, run_correct = 0.0, 0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == ytr[idx]).sum().item()
        vl, va = evaluate(model, xva, yva, batch_size)
        hist["train_loss"].append(run_loss / len(xtr))
        hist["train_acc"].append(run_correct / len(xtr))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if va > best_val:  # first epoch to reach this validation accuracy
            best_val = va
            hist["best_val_epoch"] = ep + 1
            hist["test_loss_best_val"], hist["test_acc_best_val"] = evaluate(model, xte, yte,
                                                                             batch_size)
        print(f"epoch {ep + 1:3d}/{epochs}  train loss {hist['train_loss'][-1]:.4f} "
              f"acc {hist['train_acc'][-1]:.4f} | val loss {vl:.4f} acc {va:.4f} "
              f"| {time.time() - t0:.1f}s")

    hist["test_loss"], hist["test_acc"] = evaluate(model, xte, yte, batch_size)
    return hist


def run_tag(args):
    """Results file stem. The paper baseline keeps its original tag; other nonlinearities are
    named (dann_v1_..., dann_branch12-3-1_..., dann_gate_bsupra_...), non-default constants
    add a suffix (_thr0.5_leak0.1), --apical-input coarse adds _apcoarse, --weight-decay adds _wd<value> (for any model, the baseline
    included) and --train-subset adds _nN."""
    nl = "" if args.nonlin == "leaky" else f"_{args.nonlin}"
    if args.nonlin == "branch":
        nl += args.mix.replace(":", "-")
    if args.nonlin == "gate" and args.basal_nonlin != "leaky":
        nl += f"_b{args.basal_nonlin}"
    ap = ""
    if getattr(args, "apical_input", "same") != "same":
        pool = getattr(args, "apical_pool", 4)
        ap = f"_ap{args.apical_input}" + (f"{pool}" if pool != 4 else "")
        if args.nonlin != "gate" and args.apical_frac != CONSTANT_DEFAULTS["apical_frac"]:
            ap += f"_ap{args.apical_frac:g}"  # the gate already tags its apical fraction
    return (f"dann{nl}_{args.sampling}_{args.dataset}_L{args.layers}_d{args.dends}_s{args.soma}"
            f"{constants_suffix(vars(args), args.nonlin)}{ap}"
            + (f"_wd{args.weight_decay:g}" if getattr(args, "weight_decay", 0) else "")
            + f"_seed{args.seed}"
            + (f"_n{args.train_subset}" if args.train_subset else ""))


def main():
    p = argparse.ArgumentParser(description="Chavlis & Poirazi (2025) dendritic ANN base model")
    p.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    p.add_argument("--dends", type=int, default=16, help="dendrites per soma")
    p.add_argument("--soma", type=int, default=128, help="somata per layer")
    p.add_argument("--synapses", type=int, default=16, help="inputs per dendrite")
    p.add_argument("--layers", type=int, default=1, help="dendro-somatic layers")
    p.add_argument("--sampling", default="local", choices=["random", "local", "global"],
                   help="input sampling: random (dANN-R), local RFs (dANN-LRF), global RFs (dANN-GRF)")
    p.add_argument("--nonlin", default="leaky", choices=list(DendriticANN.NONLINS),
                   help="dendritic nonlinearity: leaky (paper, LeakyReLU 0.1); v1 (three fixed "
                        "channels per dendrite: supralinear, sublinear, Ca spike); supra / sub "
                        "/ ca (homogeneous controls: that single channel on every dendrite); or "
                        "branch (one channel per dendrite, mixed within a soma by --mix)")
    p.add_argument("--mix", default="1:1:1",
                   help="supra:sub:ca dendrite ratio per soma, only with --nonlin branch "
                        "(e.g. 12:3:1 = biological, 1:1:1 = uniform)")
    g = p.add_argument_group("dendritic nonlinearity constants (v1, supra, sub, ca, branch)")
    g.add_argument("--supra-p", type=float, default=2.0, help="supra(u) = relu(u)**p, p >= 1")
    g.add_argument("--sub-sat", type=float, default=1.0, help="sub(u) = s * tanh(u / s)")
    g.add_argument("--ca-thresh", type=float, default=1.0,
                   help="ca(u) = A * sigmoid(k * (u - theta)); this is theta")
    g.add_argument("--ca-gain", type=float, default=10.0, help="k in the Ca spike sigmoid")
    g.add_argument("--ca-amp", type=float, default=1.0, help="A, the Ca plateau amplitude")
    g.add_argument("--leak", type=float, default=0.0,
                   help="passive leak added to every channel: f(u) + leak * u (0.1 = LeakyReLU's slope)")
    gg = p.add_argument_group("apical Ca gate (only used with --nonlin gate)")
    gg.add_argument("--basal-nonlin", default="leaky", choices=["leaky", "supra", "sub", "ca"],
                    help="nonlinearity of the dendrites (basal and apical) in the gate model")
    gg.add_argument("--apical-frac", type=float, default=0.25,
                    help="fraction of each soma's dendrites that are apical (0.25 of 16 = 4)")
    ga = p.add_argument_group("apical context (any --nonlin)")
    ga.add_argument("--apical-input", default="same", choices=["same", "coarse"],
                    help="what the apical dendrites (the last --apical-frac of each soma's) read: "
                         "same = local patches of the image like every dendrite (default); "
                         "coarse = an average-pooled, low-resolution copy of the image")
    ga.add_argument("--apical-pool", type=int, default=4,
                    help="pooling block size of the coarse apical view (4: 28x28 -> 7x7)")
    p.add_argument("--epochs", type=int, default=None, help="default: 25 fmnist, 50 cifar10")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam L2 weight decay on all parameters (0 = paper; run is tagged _wd<value>)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-subset", type=int, default=None,
                   help="train on the first N shuffled training images; for quick checks only "
                        "(run is tagged _nN and left out of the figures)")
    p.add_argument("--out", default="results")
    p.add_argument("--skip-existing", action="store_true",
                   help="exit at once if <out>/<tag>.json already exists (for sweep scripts)")
    args = p.parse_args()

    out = Path(args.out)
    tag = run_tag(args)
    if args.skip_existing and (out / f"{tag}.json").exists():
        print(f"skip {tag} (already done)")
        return

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    epochs = args.epochs or {"fmnist": 25, "cifar10": 50}[args.dataset]

    data, img_shape, channels, n_classes = load_data(args.dataset, args.data_dir, seed=args.seed,
                                                     train_subset=args.train_subset)
    nonlin_params = dict(supra_p=args.supra_p, sub_sat=args.sub_sat, ca_thresh=args.ca_thresh,
                         ca_gain=args.ca_gain, ca_amp=args.ca_amp, leak=args.leak)
    model = DendriticANN(img_shape, channels, n_classes, dends=args.dends, soma=args.soma,
                         synapses=args.synapses, layers=args.layers, sampling=args.sampling,
                         seed=args.seed, nonlin=args.nonlin, nonlin_params=nonlin_params,
                         mix=args.mix, basal_nonlin=args.basal_nonlin,
                         apical_frac=args.apical_frac, apical_input=args.apical_input,
                         apical_pool=args.apical_pool).to(device)

    print(f"dANN-{args.sampling} | {args.nonlin} dendrites | {args.dataset} | {args.layers} layer(s) | "
          f"{args.dends} dends x {args.soma} somata | {model.n_params():,} trainable params | {device}")
    print(f"images: train {len(data['train'][0]):,} | val {len(data['val'][0]):,} | "
          f"test {len(data['test'][0]):,}" + ("  (TRAIN SUBSET)" if args.train_subset else ""))
    hist = train(model, data, epochs, args.lr, args.batch_size, device, args.seed,
                 weight_decay=args.weight_decay)
    print(f"\nTest acc: {hist['test_acc']:.4f} | Test loss: {hist['test_loss']:.4f}"
          f" | at best-validation epoch {hist['best_val_epoch']}: {hist['test_acc_best_val']:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    hist.update(vars(args), n_params=model.n_params(), epochs=epochs)
    (out / f"{tag}.json").write_text(json.dumps(hist, indent=2))
    torch.save(model.state_dict(), out / f"{tag}.pt")
    print(f"Saved {out / tag}.json and .pt")


if __name__ == "__main__":
    main()
