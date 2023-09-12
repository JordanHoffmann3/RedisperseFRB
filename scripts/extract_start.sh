#!/bin/bash

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
for frb in "$@"
do

    cd $REDIS/Outputs/Dispersed_${frb}/job/
    echo "# DM, start pos" > start_pos.txt

    for f in slurm*
    do
        DM="$(grep 'Dispersion to DM' ${f} | awk '{print $4}')"
        idx="$(grep 'Start:' ${f} | awk '{print $2}')"
        printf "%s %s \n" ${DM} ${idx} >> temp.txt
    done

    sort -k 1 -n temp.txt >> start_pos.txt
    rm temp.txt

done
