#!/bin/bash

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
for frb in "$@"
do

    cd $REDIS/Outputs/Dispersed_${frb}/fredda_reverted/
    
    for d in $(ls -d fredda_outputs_*/)
    do
        cd $d

        if compgen -G "*.cand.fof" > /dev/null
        then
            for f in *.cand.fof
            do
                DM="$(echo $f | awk -F'[_.]' '{print $3}').$(echo $f | awk -F'[_.]' '{print $4}')"
                printf "%s \t" ${DM} >> temp.txt
                line=$(sort -k 1 -n $f | tail -1)
                echo ${line} >> temp.txt
            done

            echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > extracted_outputs.txt
            sort -k 1 -V temp.txt >> extracted_outputs.txt
            rm temp.txt
        fi

        cd $REDIS/Outputs/Dispersed_${frb}/fredda_reverted/
    done

done
