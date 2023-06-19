#!/bin/bash

source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

path=$1
cd $path

# Hi frequency
echo "# width, DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > $REDIS/Data/temp.txt

for f in *hf*.cand.fof
do
    DM="$(echo $f | awk -F'[_]' '{print $4}')"
    DM=${DM:2}
    width="$(echo $f | awk -F'[_.]' '{print $6}').$(echo $f | awk -F'[_.]' '{print $7}')"
    width=${width:5}
    sed "s/^/$width $DM /g" $f | tail -n +2 >> $REDIS/Data/temp.txt
done

sort -k1 -n -k2 -n $REDIS/Data/temp.txt > $REDIS/Data/harry_hf.txt
rm $REDIS/Data/temp.txt

# Low frequency
echo "# width, DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > $REDIS/Data/temp.txt

for f in *lf*.cand.fof
do
    DM="$(echo $f | awk -F'[_]' '{print $4}')"
    DM=${DM:2}
    width="$(echo $f | awk -F'[_.]' '{print $6}').$(echo $f | awk -F'[_.]' '{print $7}')"
    width=${width:5}
    sed "s/^/$width $DM /g" $f | tail -n +2 >> $REDIS/Data/temp.txt
done

sort -k1 -n -k2 -n $REDIS/Data/temp.txt > $REDIS/Data/harry_lf.txt
rm $REDIS/Data/temp.txt
