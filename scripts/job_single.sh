#!/bin/bash
#SBATCH --job-name=Redispersion
# SBATCH --output=$1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --mem=60G

python src/redisperse.py 230731 700.7 50.0 1271.5 1.182 --out_dir=/fred/oz002/jhoffmann/RedisperseFRB/Outputs/Dispersed_230731/outputs/ --data_dir=/fred/oz002/jhoffmann/RedisperseFRB/Outputs/Dispersed_230731/data/


