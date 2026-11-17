#!/usr/bin/env bash
# Control 1 (homogeneous type) and the per-dendrite ratio experiment: mixed V1 dendrites vs
# each single nonlinearity vs the paper baseline, and --nonlin branch at several supra:sub:ca
# ratios, over seeds and sizes. Skips any run whose results JSON already exists, so it can be
# re-run to fill in gaps. Output is appended to sweep_controls.log.
#
#   bash sweep_controls.sh                                      # seeds 1-3, 16x256, all 5 models
#   NONLINS=branch bash sweep_controls.sh                       # ratio experiment: MIXES below
#   MIXES="12:3:1 2:1:1" NONLINS=branch bash sweep_controls.sh  # other ratios
#   SEEDS="1 2 3 4 5" SIZES="16x256 64x256" bash sweep_controls.sh
#   NONLINS="ca" EXTRA="--ca-thresh 0.3" bash sweep_controls.sh # one model, extra flags
#   NONLINS=gate bash sweep_controls.sh                         # apical Ca gate model
#   (constants grids: see sweep_constants.sh)
#   EXTRA="--epochs 1 --device cpu" OUT=/tmp/dry bash sweep_controls.sh   # quick dry run
#   caffeinate -i bash sweep_controls.sh                        # keep a Mac awake
#   DATASET=cifar10 bash sweep_controls.sh                      # CIFAR-10 (see colab_cifar10.ipynb)
#
# Afterwards:
#   python plot_results.py            -> figures/7_control_homogeneous.png, control1_homogeneous.csv,
#                                        figures/8_ratio_sweep.png, ratio_sweep.csv
#   python analyze_v1.py results/dann_v1_local_fmnist_L1_d16_s256_seed1 --data-dir data
set -u
export PYTHONUNBUFFERED=1   # show each epoch line as it happens, even through tee
SEEDS=${SEEDS:-"1 2 3"}
SIZES=${SIZES:-"16x256"}                      # dendrites x somata, space separated
NONLINS=${NONLINS:-"leaky supra sub ca v1"}   # base_model.py --nonlin values (branch loops MIXES)
MIXES=${MIXES:-"12:3:1 1:1:1 1:3:12"}         # supra:sub:ca ratios for --nonlin branch
DATASET=${DATASET:-fmnist}
SAMPLING=${SAMPLING:-local}
LAYERS=${LAYERS:-1}
OUT=${OUT:-results}
EXTRA=${EXTRA:-}                              # extra base_model.py flags
PYTHON=${PYTHON:-python}
LOG=${LOG:-sweep_controls.log}

# one training run; skipped if its results JSON exists
run_one() {  # seed dends soma nonlin [mix]
  local seed=$1 d=$2 s=$3 nl=$4 mix=${5:-} tag mixflag=()
  # same tag rule as base_model.py: the leaky baseline keeps the original dann_ prefix
  case "$nl" in
    leaky)  tag="dann_${SAMPLING}" ;;
    branch) tag="dann_branch${mix//:/-}_${SAMPLING}"; mixflag=(--mix "$mix") ;;
    *)      tag="dann_${nl}_${SAMPLING}" ;;
  esac
  tag="${tag}_${DATASET}_L${LAYERS}_d${d}_s${s}_seed${seed}"
  if [ -f "$OUT/$tag.json" ]; then
    echo "skip $tag (already done)"
    return
  fi
  echo "=== seed $seed | ${d} dends x ${s} somata | $nl ${mix} ==="
  # shellcheck disable=SC2086  # EXTRA is meant to be word-split into flags
  $PYTHON base_model.py --dataset "$DATASET" --sampling "$SAMPLING" --layers "$LAYERS" \
    --nonlin "$nl" ${mixflag[@]+"${mixflag[@]}"} --dends "$d" --soma "$s" --seed "$seed" \
    --out "$OUT" --skip-existing $EXTRA  # mixflag idiom: bash 3.2 (macOS) + empty array + set -u
}

run() {
  local seed size d s nl mix
  for seed in $SEEDS; do
    for size in $SIZES; do
      d=${size%x*}; s=${size#*x}
      for nl in $NONLINS; do
        if [ "$nl" = branch ]; then
          for mix in $MIXES; do run_one "$seed" "$d" "$s" branch "$mix"; done
        else
          run_one "$seed" "$d" "$s" "$nl"
        fi
      done
    done
  done
  echo
  echo "Done. Next:"
  echo "  $PYTHON plot_results.py --results $OUT"
  echo "  $PYTHON analyze_v1.py $OUT/dann_v1_${SAMPLING}_${DATASET}_L${LAYERS}_d${d}_s${s}_seed${seed} --data-dir data"
}

run 2>&1 | tee -a "$LOG"
