#!/usr/bin/env bash
# Control 1 (homogeneous type): mixed V1 dendrites vs each single nonlinearity vs the paper
# baseline, over seeds and sizes. Skips any run whose results JSON already exists, so it can be
# re-run to fill in gaps. Output is appended to sweep_controls.log.
#
#   bash sweep_controls.sh                                      # seeds 1-3, 16x256, all 5 models
#   SEEDS="1 2 3 4 5" SIZES="16x256 64x256" bash sweep_controls.sh
#   NONLINS="ca" EXTRA="--ca-thresh 0.3" bash sweep_controls.sh # one model, extra flags
#   EXTRA="--epochs 1 --device cpu" OUT=/tmp/dry bash sweep_controls.sh   # quick dry run
#   caffeinate -i bash sweep_controls.sh                        # keep a Mac awake
#
# Afterwards:
#   python plot_results.py            -> figures/7_control_homogeneous.png, control1_homogeneous.csv
#   python analyze_v1.py results/dann_v1_local_fmnist_L1_d16_s256_seed1 --data-dir data
set -u
SEEDS=${SEEDS:-"1 2 3"}
SIZES=${SIZES:-"16x256"}                      # dendrites x somata, space separated
NONLINS=${NONLINS:-"leaky supra sub ca v1"}   # base_model.py --nonlin values
DATASET=${DATASET:-fmnist}
SAMPLING=${SAMPLING:-local}
LAYERS=${LAYERS:-1}
OUT=${OUT:-results}
EXTRA=${EXTRA:-}                              # extra base_model.py flags
PYTHON=${PYTHON:-python}
LOG=${LOG:-sweep_controls.log}

run() {
  local seed size d s nl tag
  for seed in $SEEDS; do
    for size in $SIZES; do
      d=${size%x*}; s=${size#*x}
      for nl in $NONLINS; do
        # same tag rule as base_model.py: the leaky baseline keeps the original dann_ prefix
        if [ "$nl" = leaky ]; then tag="dann_${SAMPLING}"; else tag="dann_${nl}_${SAMPLING}"; fi
        tag="${tag}_${DATASET}_L${LAYERS}_d${d}_s${s}_seed${seed}"
        if [ -f "$OUT/$tag.json" ]; then
          echo "skip $tag (already done)"
          continue
        fi
        echo "=== seed $seed | ${d} dends x ${s} somata | $nl ==="
        # shellcheck disable=SC2086  # EXTRA is meant to be word-split into flags
        $PYTHON base_model.py --dataset "$DATASET" --sampling "$SAMPLING" --layers "$LAYERS" \
          --nonlin "$nl" --dends "$d" --soma "$s" --seed "$seed" --out "$OUT" $EXTRA
      done
    done
  done
  echo
  echo "Done. Next:"
  echo "  $PYTHON plot_results.py --results $OUT"
  echo "  $PYTHON analyze_v1.py $OUT/dann_v1_${SAMPLING}_${DATASET}_L${LAYERS}_d${d}_s${s}_seed${seed} --data-dir data"
}

run 2>&1 | tee -a "$LOG"
