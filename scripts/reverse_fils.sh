#!/bin/bash

source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do
    cd $REDIS

    # Make directories
    if ! [ -d Dispersed_${frb}/reverse_fq ]
    then
        mkdir Dispersed_${frb}/reverse_fq
    fi

    cd Dispersed_${frb}/outputs

    # For each fil file
    for f in *.fil
    do
        foff=$(python $REDIS/src/flip_frequencies.py $f ../reverse_fq/$f)
        if echo "$foff < 0.0" | bc -l | grep -q 1
        then
            mv $f temp.fil
            mv ../reverse_fq/$f $f
            mv temp.fil ../reverse_fq/$f
        fi
    done
done


