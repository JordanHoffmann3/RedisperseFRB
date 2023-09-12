#!/bin/bash
#SBATCH --job-name=Redispersion
# SBATCH --output=$1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --mem=40G

cd $REDIS

