#!/usr/bin/env bash
# Full car-maze pipeline: data -> 45 runs -> summary.
# Usage: GPUS="6 7" JOBS=16 bash scripts/run_all.sh [--skip-data]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/logs
if [ "${1:-}" != "--skip-data" ]; then
  CUDA_VISIBLE_DEVICES="" $PY scripts/make_data.py   # evaluation queries, simulation-only data, paired real chains
  CUDA_VISIBLE_DEVICES="" $PY scripts/make_rows.py   # seed_R simulation data, B1/B2 windows
fi
# 3 data seeds x 5 paired sizes x 3 model seeds, JOBS at a time, spread over the listed GPUs
read -r -a G <<< "${GPUS:-6 7}"
JOBS=${JOBS:-${#G[@]}}
i=0
for d in 0 1 2; do for n in 200 400 600 800 1000; do for s in 0 1 2; do
  echo "$d $n $s ${G[$((i % ${#G[@]}))]}"; i=$((i + 1))
done; done; done | xargs -P "$JOBS" -L 1 bash -c \
  'CUDA_VISIBLE_DEVICES=$3 '"$PY"' scripts/run.py $0 $1 $2 > outputs/logs/run_d$0_n$1_s$2.log 2>&1 && echo "run $0 $1 $2 done"'
$PY scripts/report.py
