#!/bin/bash
#SBATCH --job-name=IncoherentDedispersion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --mem=16G

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
source $REDIS/loadpy.sh

for frb in "$@"
do
    cd $REDIS

    # Navigate to directories where jobs are created and executed
    cd Outputs/Dispersed_${frb}/incoherent_dedispersion

    # Create the file
    echo "" > incoherent_SNR.txt

    # Do dedispersion
    python $REDIS/src/incoherent_dedispersion.py $REDIS/Outputs/Dispersed_${frb}/outputs/ -p

    # Sort outputs
    sort -k 1 -n $(printf "incoherent_SNR.txt" ${frb}) > temp.txt
    mv temp.txt $(printf "incoherent_SNR.txt" ${frb})
done


