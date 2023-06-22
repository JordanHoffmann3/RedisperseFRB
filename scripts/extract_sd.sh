#!/bin/bash

frb=$1
DM=$2

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS/Dispersed_${frb}/sd_${DM}/
echo "# DM, sd, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > extracted_outputs.txt

for f in whole_grid/*.cand.fof
do
    sd="$(echo $f | awk -F'[_.]' '{print $6}')"
    printf "%s %s \t" ${DM} ${sd} >> temp.txt
    line=$(sort -k 1 -n $f | tail -1)
    echo ${line} >> temp.txt
done

sort -k 2 -n temp.txt >> extracted_outputs.txt
rm temp.txt

for f in channel_wise/*.cand.fof
do
    sd="$(echo $f | awk -F'[_.]' '{print $6}')"
    printf "%s %s \t" ${DM} ${sd} >> temp.txt
    line=$(sort -k 1 -n $f | tail -1)
    echo ${line} >> temp.txt
done

sort -k 2 -n temp.txt >> extracted_outputs.txt
rm temp.txt

