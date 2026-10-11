#!/usr/bin/env bash
# Full pipeline of the toy study: data -> 45 runs -> results page.
# Usage (repository root, environment from requirements.txt):
#   GPUS="0 1" JOBS=8 bash scripts/run_all.sh
# GPUS: GPU ids used round-robin; JOBS: runs executed at the same time (each needs ~1 GPU process and several GB of RAM).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/logs

CUDA_VISIBLE_DEVICES="" $PY scripts/make_data.py   # test sequences, simulation data, real chains
CUDA_VISIBLE_DEVICES="" $PY scripts/make_rows.py   # seedR simulation data, Monte Carlo rows, B1/B2 windows

# 3 data seeds x 5 paired sizes x 3 model seeds, JOBS at a time, spread over the listed GPUs
read -r -a G <<< "${GPUS:-0}"
JOBS=${JOBS:-${#G[@]}}
i=0
for d in 0 1 2; do for n in 200 400 600 800 1000; do for s in 0 1 2; do
  echo "$d $n $s ${G[$((i % ${#G[@]}))]}"; i=$((i + 1))
done; done; done | xargs -P "$JOBS" -L 1 bash -c \
  'CUDA_VISIBLE_DEVICES=$3 '"$PY"' scripts/run.py $0 $1 $2 > outputs/logs/run_d$0_n$1_s$2.log 2>&1 && echo "run $0 $1 $2 done"'

$PY scripts/report.py                              # outputs/report/report.html
