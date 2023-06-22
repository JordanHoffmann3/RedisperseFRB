#!/bin/bash

frb=$1
DM=$2

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS/Dispersed_${frb}/offset_${DM}/outputs/
echo "# DM, offset, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > extracted_outputs.txt

for f in *.cand.fof
do
    DM="$(echo $f | awk -F'[_.]' '{print $3}').$(echo $f | awk -F'[_.]' '{print $4}')"
    offset="$(echo $f | awk -F'[_.]' '{print $5}')"
    printf "%s %s \t" ${DM} ${offset} >> temp.txt
    line=$(sort -k 1 -n $f | tail -1)
    echo ${line} >> temp.txt
done

sort -k 2 -n temp.txt >> extracted_outputs.txt
rm temp.txt
