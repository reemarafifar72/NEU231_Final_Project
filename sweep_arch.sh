#!/usr/bin/env bash
# Architecture sweep for the paper's model (LeakyReLU dendrites, dANN-LRF): wider (more somata,
# more dendrites per soma) and deeper (more dendro-somatic layers) than the 1 x 16 x 256 base.
# Each ARCHS entry is LAYERSxDENDSxSOMA. Runs whose results JSON exists are skipped, so the
# sweep resumes where it stopped. Cheap architectures come first.
#
#   bash sweep_arch.sh                                   # Fashion-MNIST, seeds 1-3
#   DATASET=cifar10 bash sweep_arch.sh                   # CIFAR-10 (use Colab, see notebook 5g)
#   python arch_results.py --dataset fmnist              # table + figure
#
# Env: SEEDS (1 2 3), ARCHS (below), DATASET (fmnist), OUT (results), EXTRA, PYTHON, LOG.
set -u
export PYTHONUNBUFFERED=1
SEEDS=${SEEDS:-"1 2 3"}
DATASET=${DATASET:-fmnist}
OUT=${OUT:-results}
EXTRA=${EXTRA:-}
PYTHON=${PYTHON:-python}
LOG=${LOG:-sweep_arch_${DATASET}.log}
# base 1x16x256 and 1x16x512, 1x64x256 already exist for fmnist (5 seeds); listed so the sweep
# also covers them on a new dataset.
ARCHS=${ARCHS:-"
1x16x256 1x16x128 1x16x512 1x64x256
2x16x256 3x16x256 2x16x512 3x16x512
1x32x512 1x16x1024 2x64x256 2x16x1024
1x64x512 1x16x2048
"}

run() {
  local arch seed L d s
  for arch in $ARCHS; do
    L=${arch%%x*}; d=${arch#*x}; d=${d%x*}; s=${arch##*x}
    for seed in $SEEDS; do
      echo "=== seed $seed | $L layer(s) x $d dends x $s somata | $DATASET ==="
      # shellcheck disable=SC2086
      $PYTHON base_model.py --dataset "$DATASET" --sampling local --nonlin leaky \
        --layers "$L" --dends "$d" --soma "$s" --seed "$seed" --out "$OUT" --skip-existing $EXTRA
    done
  done
  echo "Done. Next: $PYTHON arch_results.py --dataset $DATASET"
}

run 2>&1 | tee -a "$LOG"
