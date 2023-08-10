#!/bin/bash

#SBATCH -n 1
#SBATCH -t 36:00:00
#SBATCH --mem 4096MB 
#SBATCH --tmp=4096MB
#SBATCH --job-name ANNz_Allwise
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.luken@westernsydney.edu.au
#SBATCH --output slurm_logs_allwise/annz_%A_%a.out
#SBATCH --error slurm_logs_allwise/annz_%A_%a.err

module load apptainer/latest 
export APPTAINER_BINDPATH="/fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/"

singularity run /fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/Containers/annz_latest.sif /fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/Scripts/annz_predict_allwise.py 
