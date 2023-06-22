#!/bin/bash

frb=$1
DM=$2

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS/Dispersed_${frb}/normalise_${DM}/
echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > temp.txt

for f in whole_grid/*.cand.fof
do
    printf "%s \t" ${DM} >> temp.txt
    line=$(sort -k 1 -n $f | tail -1)
    echo ${line} >> temp.txt
done

for f in channel_wise/*.cand.fof
do
    printf "%s \t" ${DM} >> temp.txt
    line=$(sort -k 1 -n $f | tail -1)
    echo ${line} >> temp.txt
done

mv temp.txt extracted_outputs.txt