# If using OzStar, the module to load is apptainer/latest. Petrichor uses singularity. 

# OzStar:
export APPTAINER_BINDPATH="/fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/"
module load apptainer

# Singularity:
# export SINGULARITY_BINDPATH="/some/location/"
# module load singularity/3.7.3

# Setup paths to scripts and containers
export container_path="/fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/Containers"
export script_path="/fred/oz237/kluken/EMU-PS/EMU_PS_Redshift/Scripts"