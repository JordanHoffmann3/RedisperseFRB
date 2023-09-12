#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

frb="$1"
DM="$2"

cd Outputs/Dispersed_${frb}/noise_${DM}

if ! [ -d fredda_outputs/ ]
then
    mkdir fredda_outputs
fi

for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp $REDIS/scripts/cudatemp.sh ${scriptname}

    echo "cd $REDIS/Outputs/Dispersed_${frb}/noise_$DM" >> ${scriptname}
    echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
    echo $runcudafdmt ${f} -t 1024 -d 8192 -o fredda_outputs/${f}.cand >> ${scriptname}
    echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
    echo python $REDIS/fredda/fredfof.py fredda_outputs/${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done

