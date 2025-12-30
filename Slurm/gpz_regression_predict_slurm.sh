#!/bin/bash

#SBATCH -t 120
#SBATCH --mem=4096MB
#SBATCH -n 4
#SBATCH --tmp=4096MB
#SBATCH --job-name GPz_Regression
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.luken@westernsydney.edu.au
#SBATCH --output slurm_logs/gpz_regres_%A_%a.out
#SBATCH --error slurm_logs/gpz_regres_%A_%a.err

source /fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/Slurm/hpc_profile_setup.sh
export APPTAINER_BINDPATH="/fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/"

singularity exec $container_path/gpz_latest.sif python3 $script_path/gmm_gpz_regression_predict.py -s 42
