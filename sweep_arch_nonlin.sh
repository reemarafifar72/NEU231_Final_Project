#!/usr/bin/env bash
# Dendritic nonlinearities on the best architecture from sweep_arch.sh. Picks the architecture
# with the highest validation accuracy (arch_results.py --print-best; never test), then trains
# Ca-only, sub-only and supra-only dendrites on it with the constants tuned earlier in the
# project (README, "Tuning the biological nonlinearities"). LeakyReLU runs already exist.
#
#   bash sweep_arch_nonlin.sh                        # after sweep_arch.sh has finished
#   WAIT=1 bash sweep_arch_nonlin.sh                 # wait for a running sweep_arch.sh first
#   ARCH=1x16x1024 bash sweep_arch_nonlin.sh         # a chosen architecture instead
#   python plot_results.py ... / arch_results.py     # see README for the comparison
#
# Env: ARCH (LAYERSxDENDSxSOMA; default: best by validation), SEEDS (1 2 3), DATASET (fmnist), OUT (results), PYTHON, EXTRA, WAIT, NOTIFY (command run with a
#      message as its argument at each milestone, e.g. a mail script), LOG.
set -u
export PYTHONUNBUFFERED=1
SEEDS=${SEEDS:-"1 2 3"}
DATASET=${DATASET:-fmnist}
OUT=${OUT:-results}
PYTHON=${PYTHON:-python}
LOG=${LOG:-sweep_arch_nonlin_${DATASET}.log}
NOTIFY=${NOTIFY:-}

GRID='leaky|
ca|--ca-thresh 0
sub|--sub-sat 0.2
supra|--supra-p 1.5 --leak 0.1'

notify() {
  echo ">>> $1"
  osascript -e "display notification \"$1\" with title \"NEU231 sweep\"" 2>/dev/null || true
  [ -n "$NOTIFY" ] && $NOTIFY "$1"
  return 0
}

run() {
  if [ -n "${WAIT:-}" ]; then
    echo "waiting for sweep_arch.sh to finish..."
    while pgrep -f "bash sweep_arch\\.sh" > /dev/null; do sleep 60; done
    $PYTHON arch_results.py --dataset "$DATASET" --results "$OUT"
    notify "Architecture sweep on $DATASET finished. Best by validation: $($PYTHON arch_results.py --dataset "$DATASET" --results "$OUT" --print-best | tr ' ' x). Starting nonlinearity runs on ${ARCH:-that architecture}."
  fi
  if [ -n "${ARCH:-}" ]; then
    read -r L D S <<< "${ARCH//x/ }"
    echo "architecture set by ARCH: $L layer(s) x $D dends x $S somata"
  else
    read -r L D S <<< "$($PYTHON arch_results.py --dataset "$DATASET" --results "$OUT" --print-best)"
    [ -z "${S:-}" ] && { echo "no best architecture found"; exit 1; }
    echo "best architecture by validation: $L layer(s) x $D dends x $S somata"
  fi
  local grid
  grid=$(mktemp)
  printf '%s\n' "$GRID" > "$grid"
  SEEDS="$SEEDS" SIZES="${D}x${S}" LAYERS="$L" DATASET="$DATASET" OUT="$OUT" PYTHON="$PYTHON" \
    EXTRA="${EXTRA:-}" GRID="$grid" LOG=/dev/null bash sweep_constants.sh
  rm -f "$grid"
  notify "Nonlinearity runs on ${L}x${D}x${S} ($DATASET) finished."
}

run 2>&1 | tee -a "$LOG"
