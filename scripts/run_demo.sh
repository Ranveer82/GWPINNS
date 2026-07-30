#!/usr/bin/env bash
# End-to-end demonstration: generate the synthetic case, fit it, score it.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-sample_data}
OUT=${OUT:-runs/demo}
ITERS=${ITERS:-6000}
LBFGS=${LBFGS:-300}
THREADS=${THREADS:-4}

echo "==> 1/3  generating synthetic case in ${DATA}"
python3 scripts/make_sample_data.py -o "${DATA}"

echo
echo "==> 2/3  fitting (${ITERS} Adam + ${LBFGS} L-BFGS iterations)"
python3 -u scripts/train.py "${DATA}/config.yaml" \
    -o "${OUT}" --iters "${ITERS}" --lbfgs "${LBFGS}" --threads "${THREADS}"

echo
echo "==> 3/3  comparing architectures"
python3 -u scripts/benchmark_architectures.py "${DATA}/config.yaml" \
    -o "${OUT}/../benchmark" --iters 1000 --threads "${THREADS}"

echo
echo "Done. Rasters, plots and report in ${OUT}"
