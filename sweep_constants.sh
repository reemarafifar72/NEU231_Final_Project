#!/usr/bin/env bash
# Constants grid for the biological nonlinearities and the Ca gate. Stage 1 runs a built-in
# grid with one seed; plot_results.py then ranks the variants by VALIDATION accuracy and prints
# the winners as grid lines; stage 2 reruns those with more seeds. Each grid line is
#     <nonlin and model flags>|<constant flags>
# and a run is skipped when its results JSON exists (base_model.py --skip-existing).
#
#   DATASET=cifar10 bash sweep_constants.sh                             # stage 1 (seed 1)
#   python plot_results.py --dataset cifar10 --select best-val          # prints winners
#   SEEDS="2 3" GRID=grid_stage2.txt DATASET=cifar10 bash sweep_constants.sh   # stage 2
#   EXTRA="--epochs 1 --device cpu" OUT=/tmp/dry bash sweep_constants.sh      # quick dry run
#
# Env: SEEDS (1), SIZES (16x256), DATASET (fmnist), SAMPLING (local), LAYERS (1), OUT (results),
#      GRID (file of grid lines; '#' comments allowed; default = the grid below), EXTRA, PYTHON, LOG.
set -u
export PYTHONUNBUFFERED=1   # show each epoch line as it happens, even through tee
SEEDS=${SEEDS:-"1"}
SIZES=${SIZES:-"16x256"}
DATASET=${DATASET:-fmnist}
SAMPLING=${SAMPLING:-local}
LAYERS=${LAYERS:-1}
OUT=${OUT:-results}
EXTRA=${EXTRA:-}
PYTHON=${PYTHON:-python}
LOG=${LOG:-sweep_constants.log}

# Default-constant runs of supra / sub / ca / v1 / branch 12:3:1 come from sweep_controls.sh.
DEFAULT_GRID='
# supra: exponent and leak
supra|--supra-p 1.5
supra|--leak 0.1
supra|--supra-p 1.5 --leak 0.1
# sub: saturation and leak
sub|--sub-sat 2
sub|--sub-sat 4
sub|--leak 0.1
sub|--sub-sat 2 --leak 0.1
sub|--sub-sat 4 --leak 0.1
# ca: threshold and leak
ca|--ca-thresh 0.5
ca|--ca-thresh 0.3
ca|--leak 0.1
ca|--ca-thresh 0.5 --leak 0.1
ca|--ca-thresh 0.3 --leak 0.1
# mixed models with the leak
v1|--leak 0.1
branch --mix 12:3:1|--leak 0.1
# apical Ca gate (default: LeakyReLU dendrites, 4 of 16 apical)
gate|
gate|--ca-thresh 0.5
gate|--apical-frac 0.5
gate --basal-nonlin supra|--leak 0.1
'

if [ -n "${GRID:-}" ]; then
  LINES=$(grep -v '^[[:space:]]*#' "$GRID")
else
  LINES=$(printf '%s' "$DEFAULT_GRID" | grep -v '^[[:space:]]*#')
fi

run() {
  local seed size d s line head flags
  for seed in $SEEDS; do
    for size in $SIZES; do
      d=${size%x*}; s=${size#*x}
      while IFS= read -r line; do
        [ -z "${line// /}" ] && continue
        head=${line%%|*}; flags=${line#*|}
        echo "=== seed $seed | ${d} dends x ${s} somata | $head $flags ==="
        # shellcheck disable=SC2086  # head, flags and EXTRA are meant to be word-split
        $PYTHON base_model.py --dataset "$DATASET" --sampling "$SAMPLING" --layers "$LAYERS" \
          --nonlin $head --dends "$d" --soma "$s" --seed "$seed" --out "$OUT" \
          --skip-existing $flags $EXTRA
      done <<< "$LINES"
    done
  done
  echo
  echo "Done. Next:"
  echo "  $PYTHON plot_results.py --results $OUT --dataset $DATASET --select best-val"
  echo "  (prints the winners as grid lines; put them in a file and rerun with SEEDS=\"2 3\" GRID=<file>)"
}

run 2>&1 | tee -a "$LOG"
