#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

frb="$1"
DM="$2"

cd Dispersed_${frb}/noise_${DM}

if ! [ -d outputs/ ]
then
    mkdir outputs
fi

for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp $REDIS/scripts/cudatemp.sh ${scriptname}

    echo "cd $REDIS/noise_$DM" >> ${scriptname}
    echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
    echo $runcudafdmt ${f} -t 1024 -d 8192 -o outputs/${f}.cand >> ${scriptname}
    echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
    echo python $REDIS/fredda/fredfof.py outputs/${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done
