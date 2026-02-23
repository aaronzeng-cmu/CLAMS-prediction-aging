#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=clams_phase_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=azeng@mit.edu

# ---------------------------------------------------------------------------
# CLAMS Phase Predictor — SLURM training script
# Partition: mit_normal · 4 CPUs · 32 GB · 12 h
# ---------------------------------------------------------------------------

CLAMS_DIR="/orcd/data/ldlewis/001/users/azeng/CLAMS_prediction_aging"
cd "$CLAMS_DIR"

# Activate conda environment
source ~/.bashrc
module load miniforge
conda activate CSFpred

# Ensure output/ exists (SBATCH --output dir must pre-exist)
mkdir -p output/

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

python scripts/train_aging_predictor.py \
    --age_group aging \
    --model_type lstm \
    --shift_ms 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 \
    --output_dir output/

echo "Job finished: $(date)"
