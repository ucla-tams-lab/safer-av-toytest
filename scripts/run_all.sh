#!/usr/bin/env bash
# Full pipeline of the toy study: data -> 45 runs -> results page.
# Usage (repository root, safer-av environment):  GPUS="0 1" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/logs

CUDA_VISIBLE_DEVICES="" $PY scripts/make_data.py   # test routes, sim data, real chains (about 1 min on 96 cores)
CUDA_VISIBLE_DEVICES="" $PY scripts/make_rows.py   # seed_R sim data, Monte Carlo rows, B1/B2 windows

# 3 data seeds x 5 paired sizes x 3 model seeds, all in parallel, spread over the listed GPUs
read -r -a G <<< "${GPUS:-0}"
i=0
for d in 0 1 2; do for n in 200 400 600 800 1000; do for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $PY scripts/run.py $d $n $s > outputs/logs/run_d${d}_n${n}_s${s}.log 2>&1 &
  i=$((i + 1))
done; done; done
wait

$PY scripts/report.py                              # outputs/report/report.html
