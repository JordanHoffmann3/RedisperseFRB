#!/bin/bash

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do
    cd $REDIS

    # Make directories
    if ! [ -d Dispersed_${frb}_fix_fch1/ ]
    then
        mkdir Dispersed_${frb}_fix_fch1/
        mkdir Dispersed_${frb}_fix_fch1/outputs
    fi

    cd Dispersed_${frb}/outputs

    # For each fil file
    for f in *.fil
    do
        python $REDIS/src/change_fch1.py $f $REDIS/Dispersed_${frb}_fix_fch1/outputs/$f
    done
done


