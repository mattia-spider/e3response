#!/bin/bash
# Direct vs Sobolev comparison with k-fold cross-validation: one array task per
# (model, size, fold, seed).  Every run uses the same data, test set, loss weights and seed
# handling; only the model differs.  DATASET below picks the experiment:
#
#   bto        Lorenzo's BTO (Born, Raman, dielectric)       experiment=bto_sobolev_vs_direct
#   allegro    allegro-pol BaTiO3 (E, F, d, Born, χ)         experiment=bto_pol_sobolev_vs_direct
#
#   DATASET=allegro sbatch --array=0-$(( $(DATASET=allegro scripts/cluster/kfold.sbatch --count) - 1 )) scripts/cluster/kfold.sbatch
#
# Check the number of tasks with `scripts/cluster/kfold.sbatch --count` and the run of one task
# with `scripts/cluster/kfold.sbatch --show 7` (both honour DATASET).
#
#SBATCH --job-name=sobolev-kfold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
# --- site specific: fill in before submitting -----------------------------------------------
##SBATCH --account=<project>@<gpu-partition>
##SBATCH --partition=<partition>
##SBATCH --qos=<qos>
# ----------------------------------------------------------------------------------------------

set -euo pipefail

# ---- grid -------------------------------------------------------------------------------------
DATASET=${DATASET:-bto}
case "$DATASET" in
  bto)     EXPERIMENT=bto_sobolev_vs_direct;     DATA=bto_kfold
           MODELS=(nequip_bto_direct nequip_bto_sobolev)
           SIZES=("++mul=8 ++ell_max=3 ++num_layers=3") ;;     # Raman needs l = 3 in the direct heads
  allegro) EXPERIMENT=bto_pol_sobolev_vs_direct; DATA=bto_allegro_pol_kfold
           MODELS=(nequip_bto_pol_direct nequip_bto_pol_sobolev)
           SIZES=("++mul=8 ++ell_max=2 ++num_layers=3") ;;
  *) echo "unknown DATASET=$DATASET (bto or allegro)" >&2; exit 1 ;;
esac
FOLDS=(0 1 2 3 4)                         # must match n_folds in configs/data/*_kfold.yaml
SEEDS=(0)
BATCH_SIZE=4                              # from scripts/probe_batch_size.py on this GPU type
EXTRA=""                                  # further overrides shared by every run
OUT_ROOT="${SCRATCH:-$PWD}/e3response_runs/${EXPERIMENT}"
# ------------------------------------------------------------------------------------------------

n_models=${#MODELS[@]}; n_sizes=${#SIZES[@]}; n_folds=${#FOLDS[@]}; n_seeds=${#SEEDS[@]}
total=$(( n_models * n_sizes * n_folds * n_seeds ))
if [[ "${1:-}" == "--count" ]]; then echo "$total"; exit 0; fi

task=${SLURM_ARRAY_TASK_ID:-0}
if [[ "${1:-}" == "--show" ]]; then task=$2; fi
if (( task >= total )); then echo "task $task out of range (0..$((total - 1)))" >&2; exit 1; fi

# task -> (model, size, fold, seed), model varying slowest
i=$task
seed=${SEEDS[$(( i % n_seeds ))]}; i=$(( i / n_seeds ))
fold=${FOLDS[$(( i % n_folds ))]}; i=$(( i / n_folds ))
size=${SIZES[$(( i % n_sizes ))]}; i=$(( i / n_sizes ))
model=${MODELS[$i]}

tag="${model}/$(echo "$size" | tr ' =' '_-')/fold${fold}_seed${seed}"
cmd=(e3response train -i configs/train.yaml
     experiment="$EXPERIMENT" data="$DATA" model="$model"
     data.fold="$fold" seed="$seed" ++data.batch_size="$BATCH_SIZE"
     $size $EXTRA
     trainer=gpu logger=csv extras.print_config=false
     paths.root_dir="$OUT_ROOT/$tag" paths.data_dir="$PWD/data")

if [[ "${1:-}" == "--show" ]]; then echo "task $task: ${cmd[*]}"; exit 0; fi

# --- site specific environment: fill in ----------------------------------------------------------
# module purge; module load <cuda/python modules>
# source <conda>/etc/profile.d/conda.sh && conda activate <env with reax 0.7, tensorial nmr-dev>
# --------------------------------------------------------------------------------------------------

# Allocate GPU memory as needed rather than 75% up front, as in scripts/probe_batch_size.py
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$OUT_ROOT/$tag"
echo "task $task on $(hostname): ${cmd[*]}"
"${cmd[@]}"
