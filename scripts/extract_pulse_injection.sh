#!/bin/bash

source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

frb=$1

cd $REDIS/Dispersed_$frb/pulse_injection/

echo "# width, DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > temp.txt

for f in *.cand.fof
do
    DM="$(echo $f | awk -F'[_.]' '{print $3}').$(echo $f | awk -F'[_.]' '{print $4}')"
    DM=${DM:2}
    width="$(echo $f | awk -F'[_.]' '{print $5}').$(echo $f | awk -F'[_.]' '{print $6}')"
    width=${width:5}
    sed "s/^/$width $DM /g" $f | tail -n +2 >> temp.txt
done

sort -k1 -n -k2 -n temp.txt > extracted_outputs.txt
rm temp.txt
