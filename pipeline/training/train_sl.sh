#!/bin/bash
#SBATCH --account=<your_project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --job-name=fieldmark_sl
#SBATCH --output=fieldmark_sl-%j.log


module purge
module use /appl/local/csc/modulefiles
module load pytorch


cd "$SLURM_SUBMIT_DIR"
python3 train_sl.py