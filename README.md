# NEU231_Final_Project

To run the repo model --

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

python base_model.py --dataset fmnist --dends 16 --soma 128 --sampling local

## Results and conclusions

**In short:** the replication matches the paper, and biologically inspired dendritic
nonlinearities do not meaningfully improve on its LeakyReLU dendrite. The one variant that
wins is a steep saturating dendrite, by about one point at the paper's model size; the gain
disappears at a larger size and is not specific to calcium spikes. The nonlinearity matters
more for robustness: steep saturating dendrites are less tolerant of pixel noise on both
datasets, and supralinear ones more tolerant. The change that helps most is in the wiring,
not the nonlinearity: letting each soma's apical dendrites read a low-resolution, wide-field
copy of the image raises CIFAR-10 accuracy by about one point at equal parameter count and
makes the model more noise-tolerant on both datasets. It works with the plain additive soma;
the multiplicative Ca gate does not benefit from it. The two one-point gains add: steep
saturating dendrites with coarse apical input reach 59.4 to 59.7% on CIFAR-10, about two
points above the paper's model with the same number of parameters (3 seeds, one model size).

All numbers are test accuracy (%) for dANN-LRF, mean +/- SD over seeds, at 16 dendrites x 256
somata unless stated. "Paired" differences compare runs with the same seed (same data split
and wiring). Tables: `figures/cifar10/` and `figures/control1/` (Fashion-MNIST), in
`summary.csv`, `constants_sweep.csv` and `paired_vs_baseline.csv`.

| Model | CIFAR-10 | Seeds | Paired difference from baseline |
|---|---|---|---|
| LeakyReLU (paper's model; paper reports 56.97 +/- 0.98) | 57.32 +/- 0.85 | 5 | |
| Ca-only, threshold 0 | 58.22 +/- 0.91 | 5 | +0.90 +/- 0.59, p = 0.03, 4 of 5 seeds |
| Matched tanh (`sub`, saturation 0.2) | 58.55 +/- 0.39 | 3 | +1.12 +/- 1.09 |
| V1 mixed, threshold 0 | 58.57 +/- 0.50 | 3 | +1.14 +/- 1.33 |
| LeakyReLU, coarse apical input (`--apical-input coarse`) | 58.22 +/- 0.93 | 5 | +0.90 +/- 0.37, p = 0.006, 5 of 5 seeds |
| Matched tanh + coarse apical input | 59.68 +/- 0.87 | 3 | +2.25 +/- 0.37, p = 0.009, 3 of 3 seeds |
| Ca-only, threshold 0 + coarse apical input | 59.39 +/- 0.46 | 3 | +1.97 +/- 0.89, p = 0.06, 3 of 3 seeds |
| Ca gate | 57.56 +/- 0.64 | 3 | +0.13 +/- 0.32 |
| Ca gate, coarse apical input | 57.31 +/- 0.99 | 3 | -0.12 +/- 0.39 |
| supra-only, tuned (p 1.5, leak 0.1) | 56.72 +/- 0.31 | 3 | -0.71 +/- 1.12 |
| Ca-only, default threshold 1.0 | 55.50 +/- 0.23 | 3 | -1.92 +/- 1.11 |
| Per-dendrite mix 12:3:1 (biological), threshold 0.3 | 56.84 +/- 1.08 | 3 | -0.59 +/- 0.42 |
| Per-dendrite mix 1:3:12 (Ca-heavy), threshold 0.3 | 57.72 +/- 0.46 | 3 | +0.30 +/- 0.87 |

1. **Replication.** The LeakyReLU baseline reproduces the paper on CIFAR-10. On Fashion-MNIST
   it reaches 89.13 +/- 0.70 (3 seeds).
2. **Default constants fail.** With the initial constants every biological variant is at or
   below the baseline. For Ca the cause is the threshold: at 1.0, 42% of Ca dendrites never
   switch on (Fashion-MNIST); at 0, 8% do not.
3. **A steep saturating dendrite wins at 16 x 256.** Ca-only at threshold 0 beats the baseline
   on CIFAR-10 (table above) and on Fashion-MNIST (89.70 vs 89.13, paired +0.57 +/- 0.10,
   p = 0.01, 3 of 3 seeds), at equal parameter count.
4. **It is not specific to Ca spikes.** A tanh dendrite matched to the same curve ties Ca-only
   (-0.13 pp on the same three seeds). Steepness matters for training: gain 10 is a sweet
   spot, and gains 1, 3 and 30 give 55.05, 57.03 and 54.58.
5. **It does not hold at every size.** The paired difference is +0.49 +/- 0.77 at 8 x 128,
   +0.90 at 16 x 256 and -0.13 +/- 0.51 at 32 x 256 (3 seeds at the outer sizes). At 32 x 256
   Ca-only overfits: 83% train accuracy against 74% for the baseline.
6. **The gain survives weight decay.** With decay 1e-4 the paired difference is +1.31 +/- 0.78
   and with 1e-3 it is +2.95 +/- 0.59 (3 of 3 seeds each), though decay lowers both models'
   accuracy. Early stopping changes little at 16 x 256.
7. **Mixing types adds nothing.** V1 mixed is level with the best single type on CIFAR-10 and
   ties the baseline on Fashion-MNIST (89.24). Among per-dendrite mixes, more Ca dendrites is
   better and the biological 12:3:1 ratio is the worst.
8. **The Ca gate is a gain switch, not a coincidence detector.** It gives no advantage; gate
   values are bimodal and nearly unrelated to basal drive (r = -0.19).
9. **Trained Ca dendrites are selective.** On CIFAR-10, 78% of outputs sit near 0 or 1 and 85%
   of dendrites switch between images. On Fashion-MNIST they are more graded (45% near 0 or 1).

10. **Dendrite type matters for robustness, and it cuts both ways.** Models trained on clean
    images, tested on corrupted ones; paired difference from the baseline in pp (3 seeds; 5 for
    Ca-only on CIFAR-10). Figures: `13_robustness.png` in `figures/control1/` and
    `figures/cifar10/`.

    Fashion-MNIST:

    | Model | Noise SD 0.2 | Noise SD 0.3 | 25% occluded | 40% occluded |
    |---|---|---|---|---|
    | supra-only | +9.5 | +10.9 | +0.8 | +1.0 |
    | Ca-only, threshold 0 | -9.4 | -5.9 | +3.1 | +4.1 |
    | Matched tanh | -8.3 | -3.5 | +3.3 | +4.4 |
    | Additive soma, coarse apical input | +2.6 | +2.3 | 0.0 | +1.3 |
    | Ca gate | -3.0 | -3.4 | -1.5 | -1.5 |
    | Ca gate, coarse apical input | +2.4 | +1.7 | +0.5 | +1.2 |

    CIFAR-10:

    | Model | Noise SD 0.2 | Noise SD 0.3 | 25% occluded | 40% occluded |
    |---|---|---|---|---|
    | supra-only | +0.4 | +1.2 | +0.1 | +0.5 |
    | Ca-only, threshold 0 | -2.3 | -4.2 | +0.3 | -0.4 |
    | Matched tanh | -1.0 | -1.8 | -0.4 | -1.9 |
    | V1 mixed, threshold 0 | -1.0 | -2.4 | +1.2 | +0.5 |
    | Additive soma, coarse apical input (5 seeds) | +2.1 | +2.7 | +0.1 | -0.4 |
    | Matched tanh + coarse apical input | +1.7 | +1.8 | +0.1 | -1.2 |
    | Ca-only, threshold 0 + coarse apical input | -0.2 | -0.7 | +0.8 | -0.3 |
    | Ca gate | -0.3 | -0.5 | +0.2 | -0.4 |
    | Ca gate, coarse apical input | -0.4 | -0.2 | -0.5 | -0.9 |

    What holds on both datasets: steep saturating dendrites (Ca-only and the matched tanh,
    which again behave as one function) lose accuracy under pixel noise (0 of 5 and 0 of 3
    seeds above the baseline at SD 0.2 and above on CIFAR-10), and supralinear dendrites gain
    under strong noise (3 of 3 seeds at SD 0.3 on both), though the size of both effects is far
    smaller on CIFAR-10. What does not hold: the occlusion advantage of steep saturating
    dendrites on Fashion-MNIST is absent on CIFAR-10. So Ca-only's one-point gain on clean
    CIFAR-10 images turns into a loss once noise reaches SD 0.2.
11. **No data-efficiency advantage.** Trained on 1,000, 5,000 or 20,000 images, the steep
    saturating dendrites keep roughly the edge they have on the full training set and it does
    not grow as data shrinks (`14_data_efficiency.png`). Fashion-MNIST: +0.2 to +0.5 pp, 3 of
    3 seeds at every size. CIFAR-10: matched tanh +0.93, +0.58, +0.89 (3 of 3 seeds at each
    size) against +1.12 on all 45,000; Ca-only +1.07, -0.07, +0.62 against +0.90.
12. **Coarse apical context helps the additive soma, not the gate** (3 seeds; 5 for the
    additive soma on CIFAR-10; `15_apical_context.png`, `apical_context.csv`).

    | | Fashion-MNIST, local | Fashion-MNIST, coarse | CIFAR-10, local | CIFAR-10, coarse |
    |---|---|---|---|---|
    | Additive soma (LeakyReLU) | 89.13 | 89.08 | 57.32 | 58.22 |
    | Ca gate | 89.43 | 89.24 | 57.56 | 57.31 |

    On CIFAR-10 the paper's model with coarse apical input beats the paper's model by
    +0.90 +/- 0.37 pp (p = 0.006, 5 of 5 seeds) with the same number of parameters, and the
    margin grows under noise: +1.2 at SD 0.1, +2.1 at 0.2, +2.7 at 0.3 and at 0.5, 5 of 5
    seeds at every level. The same noise benefit appears on Fashion-MNIST, where clean accuracy
    is at ceiling and unchanged. It does not help with occlusion. The gate gets nothing from
    context: its context effect is -0.25 +/- 0.40 on CIFAR-10, and the interaction (does the
    gate gain more from context than the additive soma?) is -1.33 +/- 0.70 (0 of 3 seeds,
    p = 0.08) on CIFAR-10 and -0.14 +/- 0.22 on Fashion-MNIST. So the benefit comes from giving
    the cell a second, wide-field view of the input, summed linearly at the soma; multiplying
    by a Ca plateau throws it away. Note that this is a change of input wiring, close to a
    two-scale input, rather than a dendritic nonlinearity, and that it was tested at one model
    size.

    Gate ablation (Fashion-MNIST only; not obtained for CIFAR-10): shuffling the gate across
    images costs the local-input gate 2.6 to 4.5 pp but the coarse-input gate only 0.7 to
    1.4 pp, so the network relies on the gate less when it carries context. It also shows the
    gate is not purely a gain switch on Fashion-MNIST: its values are image-specific even
    though they are uncorrelated with basal drive.

13. **The two gains add** (CIFAR-10, 3 seeds, same figure and table). Steep saturating
    dendrites combined with coarse apical input:

    | Dendrite | Local apical input | Coarse apical input | Both changes vs the paper's model |
    |---|---|---|---|
    | LeakyReLU (paper) | 57.32 | 58.22 | +0.90 +/- 0.37 (5 of 5 seeds) |
    | Ca-only, threshold 0 | 58.22 | 59.39 | +1.97 +/- 0.89 (p = 0.06, 3 of 3) |
    | Matched tanh | 58.55 | 59.68 | +2.25 +/- 0.37 (p = 0.009, 3 of 3) |

    The interaction between dendrite type and context is near zero (-0.37 +/- 0.65 for
    Ca-only, +0.05 +/- 1.17 for the matched tanh), so each change contributes its own point.
    Coarse context also offsets the noise fragility of steep dendrites (finding 10): the
    matched tanh with coarse input is +1.7 to +2.5 pp above the baseline at every noise level
    (3 of 3 seeds), and Ca-only with coarse input is level with the baseline under noise where
    Ca-only alone is 2 to 4 pp below it. This is the best model in the project: 59.68 +/- 0.87
    against 57.32 +/- 0.85 with identical parameter count. It is 3 seeds at 16 x 256 only; the
    steep-dendrite gain vanished at 32 x 256 (finding 5), and the combination has not been
    tested there or on Fashion-MNIST.

14. **Width beats depth, and size beats the dendrite type** (Fashion-MNIST, LeakyReLU dendrites,
    3 seeds; `figures/architecture/fmnist/`). Architectures are layers x dendrites x somata.

    | Architecture | Params | Test accuracy | Paired difference from 1 x 16 x 256 |
    |---|---|---|---|
    | 1 x 16 x 2048 | 612k | 90.73 +/- 0.20 | +1.60 +/- 0.71 (3 of 3) |
    | 1 x 64 x 512 | 595k | 90.53 +/- 0.12 | +1.40 +/- 0.59 (3 of 3) |
    | 1 x 16 x 1024 | 306k | 90.52 +/- 0.26 | +1.39 +/- 0.48 (3 of 3) |
    | 2 x 16 x 1024 | 602k | 90.38 +/- 0.15 | +1.25 +/- 0.59 (3 of 3) |
    | 2 x 16 x 512 | 301k | 90.31 +/- 0.28 | +1.18 +/- 0.83 (3 of 3) |
    | 1 x 32 x 512 | 301k | 90.29 +/- 0.19 | +1.16 +/- 0.66 (3 of 3) |
    | 3 x 16 x 512 | 449k | 89.84 +/- 0.73 | +0.71 +/- 0.50 (3 of 3) |
    | 1 x 16 x 256 (base) | 77k | 89.23 +/- 0.53 (5 seeds) | |

    Accuracy rises with size up to about 300k parameters (+1.2 to +1.4 pp) and barely after
    (doubling again adds 0 to 0.2 pp). At matched parameter count one layer is as good as or
    better than two, and three layers are worse; more somata and more dendrites per soma are
    interchangeable. The train-test gap grows from 3.4 to 5.5 pp with size. The top five
    architectures are within 0.15 pp on validation accuracy, so there is no single best one;
    1 x 16 x 1024 (the base with 4x the somata) was taken for the next test.

    On 1 x 16 x 1024 the dendrite type no longer matters: paired differences from LeakyReLU
    (90.52) are +0.08 +/- 0.28 for Ca-only threshold 0 (90.60), +0.09 +/- 0.19 for the matched
    tanh (90.60, 2 of 3 seeds each) and -0.53 +/- 0.33 for tuned supra-only (89.98, 0 of 3).
    At 1 x 16 x 256 the same Ca and tanh dendrites gain +0.57 and +0.48 (3 of 3). As on CIFAR-10
    at 32 x 256 (finding 5), the steep-dendrite advantage is a small-model effect that a wider
    network absorbs. CIFAR-10 has not been run for this sweep yet.

Caveats: most comparisons use 3 seeds, the p-values are uncorrected for the number of models
compared, and the biological variants were tuned while the baseline was tuned only for weight
decay.

## V1-style dendritic nonlinearities

`--nonlin v1` replaces each dendrite's LeakyReLU with three fixed nonlinearities applied in
parallel (supralinear `relu(u)^p`, sublinear `s*tanh(u/s)`, calcium spike
`A*sigmoid(k*(u-theta))`). Each soma then learns a weighted sum over all three channels of
each of its own dendrites. The default `--nonlin leaky` is the paper's model, unchanged.

python base_model.py --nonlin v1 --dends 16 --soma 256            # defaults: p=2, s=1, theta=1, k=10, A=1
python base_model.py --nonlin v1 --ca-thresh 0.5 --supra-p 1.5    # change the fixed constants

Results are tagged `dann_v1_*`; `plot_results.py` draws them next to the baseline (dashed
lines, square markers). `analyze_v1.py results/dann_v1_... --data-dir data` reports which
channel each soma relies on and whether dendrites reach the Ca threshold.

## Control 1: homogeneous type ("any single nonlinearity is enough")

`--nonlin supra`, `--nonlin sub` and `--nonlin ca` give every dendrite just that one
nonlinearity, with the baseline's cable layer and parameter count. Together with the LeakyReLU
baseline they are the single-nonlinearity controls for the mixed `v1` model.

bash sweep_controls.sh                      # seeds 1-3 at 16x256 for leaky, supra, sub, ca, v1
SEEDS="1 2 3 4 5" SIZES="16x256 64x256" bash sweep_controls.sh
python plot_results.py                      # figures/7_control_homogeneous.png + control1_homogeneous.csv

The CSV compares the mixed model with the best single nonlinearity at each size and gives a
verdict: "mixed wins" if the gap exceeds two combined SDs (and 0.5 pp), else "single is enough".

### Ratio experiment: one nonlinearity per dendrite

`--nonlin branch --mix S:B:C` gives every dendrite a single nonlinearity, with the ratio
setting how many of each soma's dendrites are supra, sub and Ca (12:3:1 at 16 dendrites is
exactly 12, 3 and 1). The cable layer and parameter count are the baseline's, so this tests
mixing across branches without the extra cable weights of `v1`.

Default ratios: `12:3:1` (biological: pyramidal dendrites are mostly NMDA-spike branches,
sublinear integration is a minority regime, and the Ca spike is one apical site per cell),
`1:1:1` (uniform), `1:3:12` (Ca-heavy counterfactual).

NONLINS=branch bash sweep_controls.sh        # the three ratios, seeds 1-3 at 16x256 (~25 min on MPS)
python plot_results.py                       # figures/8_ratio_sweep.png + ratio_sweep.csv

Figure 8 shows each ratio with error bars against the best single nonlinearity and the V1
mixed model at the same size; the CSV adds a "ratio wins" / "no advantage" verdict.

## Tuning the biological nonlinearities

The fixed constants (Ca threshold, gain, exponent, saturation) were initial guesses, and the
single-nonlinearity controls lose about a point to LeakyReLU on CIFAR-10 partly because the
fixed functions clip: Ca is bounded and flat away from threshold, tanh saturates, relu^p is
zero for negative input. Two remedies:

- `--leak L` adds a passive component to every channel, `f(u) + L*u` (real dendrites conduct
  passively alongside any active mechanism; 0.1 is the slope LeakyReLU keeps). Default 0.
- `sweep_constants.sh` runs a grid over the constants (and the Ca gate below) with one seed.
  `python plot_results.py --select best-val` ranks the variants by **validation** accuracy,
  never test, prints the winners as grid lines, and uses them in the comparison figures. Rerun
  the winners with `SEEDS="2 3" GRID=<file>` before drawing conclusions.

Runs with non-default constants get a tag suffix (`..._thr0.5_leak0.1_seed1`) and appear in
the figures as `model [thr0.5_leak0.1]`. `9_constants_sweep.png` / `constants_sweep.csv` show
every variant.

## Ca gate model (`--nonlin gate`)

The Ca spike's job in a pyramidal cell is not to add to the somatic input but to detect
coincidence between apical and basal drive and turn a spike into a burst. The gate model gives
each soma that computation, which no weighted sum of fixed functions can express:

    B  = basal dendrites' drive        (12 of 16 dendrites, LeakyReLU by default)
    A  = apical dendrites' drive        (4 of 16)
    Ca = sigmoid(k (A - theta))         apical plateau in [0, 1]
    soma = LeakyReLU( B * (1 + g Ca) )  apical gates basal

Flags: `--apical-frac` (0.25), `--basal-nonlin {leaky,supra,sub,ca}` (leaky, so a win is
attributable to the gate), `--ca-thresh`, `--ca-gain`, `--ca-amp`. Masked weights equal the
baseline's; there is one extra bias vector. `analyze_v1.py` reports how often the gate opens
and the correlation between basal drive and gate (`10_gate_activity.png`). Note that theta is
applied to the learned apical sum, so the apical bias can shift the effective threshold.

## Follow-up experiments: is the Ca-only result real, and why?

With a working threshold (`--ca-thresh 0`), Ca-only dendrites beat the LeakyReLU baseline on
CIFAR-10 at 16 x 256. Four checks (outcomes are in "Results and conclusions" above), all run through `sweep_constants.sh` grids (section 5e of the
Colab notebook has the cells):

- **Other sizes.** `SIZES="8x128 32x256" SEEDS="1 2 3"` with grid lines `leaky|` and
  `ca|--ca-thresh 0`. `11_size_generalisation.png` shows accuracy by size and the paired
  difference from the baseline at each size.
- **Steepness or just a sigmoid?** At threshold 0 the Ca function is `sigmoid(gain * u)`, and
  the synaptic weights can absorb the gain, so `--ca-gain 1 / 3 / 30` only tests sensitivity to
  the initial steepness. The matched control is `--nonlin sub --sub-sat 0.2`:
  `0.2 tanh(5u) = 0.4 sigmoid(10u) - 0.2`, the same curve up to a shift and scale the soma
  absorbs. It does match Ca-only, so the result is about steep bounded dendrites, not Ca
  spikes specifically.
- **Regularisation.** `--weight-decay W` is Adam's L2 penalty on every parameter, biases
  included (runs are tagged `_wd<W>`; default 0 = the paper). Every run now also saves
  `test_acc_best_val`, the test accuracy at its best-validation epoch (early stopping chosen on
  validation, never on test). `test_acc` is still the final epoch.
- **What the dendrites do.** `analyze_v1.py <ca run> --data-dir data` reports how often Ca
  dendrites sit near 0 or 1 and how many are selective, stuck off or stuck on
  (`12_dendrite_states.png`).

`plot_results.py` writes `paired_vs_baseline.csv`: each model minus the baseline on the seeds
they share (same seed = same data split and wiring), with the SD of the differences, a paired
t-test and the number of seeds won. Decayed models are paired with the equally decayed
baseline. The p-values are not corrected for the number of models compared; with 3 to 5 seeds
they are suggestive, not conclusive.

## Beyond clean accuracy: robustness, data efficiency and apical context

Clean test accuracy barely separates the models, so three further experiments ask where
dendritic nonlinearities might matter instead (Colab notebook section 5f has the CIFAR-10 cells).

**Robustness.** `robustness.py` trains nothing: it evaluates every 16 x 256 run that has
saved weights on corrupted test images, Gaussian pixel noise (SD 0.05 to 0.5) and a grey
square covering 10 to 40% of the image. The corruption is generated once from a fixed seed
and is the same for every model. Output: `13_robustness.png`, `robustness.csv` and
`robustness_paired.csv` (each model minus the baseline with the same seed, per level).

    python robustness.py --dataset fmnist --out figures/control1

**Data efficiency.** Baseline, Ca-only threshold 0 and the matched tanh trained on 1,000,
5,000 and 20,000 images (`--train-subset N`, with 200, 100 and 50 epochs so small subsets are
not simply under-trained) and compared at each run's best-validation epoch.
`plot_results.py` draws `14_data_efficiency.png` and writes `data_efficiency.csv`; subset runs
stay out of every other table.

    GRID=grid.txt SEEDS="1 2 3" EXTRA="--train-subset 1000 --epochs 200" bash sweep_constants.sh

**Apical context (`--apical-input coarse`).** In the Ca gate model the apical and basal
dendrites both read local patches of the same image, so the gate has no second signal to
detect a coincidence with. With `--apical-input coarse`, each soma's apical dendrites (the
last `--apical-frac` of its dendrites, 4 of 16) take their synapses from a 4x average-pooled
copy of the image (`--apical-pool`), a low-resolution wide-field view like the contextual
input the apical tuft receives in cortex. Basal dendrites keep their local receptive fields
and their wiring for a given seed; pooling has no parameters and every dendrite keeps 16
synapses, so the parameter count is unchanged. (The first layer's input is longer, so its
initial weights are not identical to the baseline's for the same seed.) Runs are tagged
`_apcoarse`.

The flag works with any `--nonlin`, which gives a 2 x 2 design: additive soma (`leaky`) or
multiplicative gate (`gate`), each with local or coarse apical input. Coarse context helping
both equally would mean the benefit is the extra view; the gate idea is supported only if
gate + context beats additive + context. `plot_results.py` writes `apical_context.csv` with
the four cells, the paired contrasts and their interaction, and `15_apical_context.png`.
These runs are kept out of the comparisons with the paper.

Because the flag is independent of the dendrite type, the same table also tests whether the
two gains add: Ca-only threshold 0 and the matched tanh, each with coarse apical input
(notebook cell 5f D), next to the same dendrites with local input and to the LeakyReLU models.

    python base_model.py --dataset cifar10 --nonlin sub --sub-sat 0.2 --apical-input coarse

`apical_context.csv` then also holds, per dendrite type, the context effect, the dendrite
effect under coarse input, the combination against the paper's model, and the interaction
(dendrite type x context; zero means the gains add). The same lines are drawn in
`15_apical_context.png`.

For gate runs `analyze_v1.py` now
also reports a gate ablation: test accuracy with the gate shuffled across images (same values,
wrong images), fixed at each soma's mean, and closed. Intact minus shuffled is the
image-specific information the gate carries.

## Architecture sweep: wider and deeper models

`sweep_arch.sh` trains the paper's model (LeakyReLU, dANN-LRF, 16 synapses) at 14
architectures, `LAYERSxDENDSxSOMA`, seeds 1-3, cheapest first and resumable. Extra layers
read the previous layer's somata laid out on a 2D grid, as in the paper's code.
`arch_results.py` writes `arch_summary.csv` (final and early-stopped test accuracy, train-test
gap, paired difference from 1 x 16 x 256) and `16_architecture.png`.
`sweep_arch_nonlin.sh` then trains Ca-only (threshold 0), the matched tanh (`sub`, saturation
0.2) and tuned supra-only (p 1.5, leak 0.1) on one architecture: the best by validation
accuracy (`arch_results.py --print-best`, never test) or the one given as `ARCH`.

    bash sweep_arch.sh                                   # Fashion-MNIST, about 6 h on an M2
    python arch_results.py --dataset fmnist              # figures/architecture/fmnist/
    ARCH=1x16x1024 bash sweep_arch_nonlin.sh             # about 1.5 h
    ARCHS="1x16x256 1x16x1024" DATASET=cifar10 bash sweep_arch.sh   # Colab, notebook 5g

## CIFAR-10 on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HillaryKchao/NEU231_Final_Project/blob/bio-tuning/colab_cifar10.ipynb)

CIFAR-10 is a single 160 MB archive and each epoch reads all of it, so training needs the
dataset on the machine doing the work. `colab_cifar10.ipynb` runs the whole comparison on a
free Colab GPU instead: the dataset lands on Colab's disk, not yours, and the sweep takes a
few hours rather than 10+ on a laptop.

1. Open the notebook with the badge, pick a GPU runtime, run the cells top to bottom.
2. It clones this branch, smoke-tests one epoch, runs control 1 and the ratio sweep with
   `DATASET=cifar10`, and makes the figures in `figures/cifar10/`.
3. The last cell downloads a zip of the result JSONs, figures and log. Unzip into `results/`
   here and run `python plot_results.py --dataset cifar10 --out figures/cifar10` to regenerate.

Free sessions can time out; the sweep skips finished runs, so re-running the cells resumes.
Set `USE_DRIVE = True` in the notebook to keep results in Google Drive across sessions.

`--train-subset N` trains on the first N shuffled training images (validation and test stay
full). It is for quick checks: such runs are tagged `_nN` and `plot_results.py` leaves them out.
