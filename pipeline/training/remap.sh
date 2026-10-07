#!/bin/bash
# remap.sh — run the label remap (coarse -> species classes) on a COMPUTE node.
# Do the instant dir swaps on the login node first, then:  sbatch remap.sh
#SBATCH --account=<your_project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=remap
#SBATCH --output=remap-%j.log

module purge
module use /appl/local/csc/modulefiles
module load pytorch

cd "$SLURM_SUBMIT_DIR"
python3 remap.py
