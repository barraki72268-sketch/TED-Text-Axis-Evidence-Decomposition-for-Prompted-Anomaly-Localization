#!/bin/bash
#SBATCH --job-name=ted-bayes-fresh-bank-v2
#SBATCH --partition=gpu
#SBATCH --reservation=cal_jinyoungkim_rtx6000ada_833b0d274e93
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --dependency=afterany:13798
#SBATCH --output=/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/reports/bayes-fresh-bank-v2-%j.log
set -euo pipefail
cd /mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/github-fresh-bank-20261010-v1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /mnt/data/envs/patchcore-conda/bin/python -u -m reproduction.bayes_bank_build ../rtx-bayes-runs-20261010/bayespfl-vitb_plus-mvtec2btad-seed0 ../fresh-banks/bayes-bplus-mvtec-seed0-gpu-20261010-v2
