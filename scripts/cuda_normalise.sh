#!/bin/bash
source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

frb="$1"
DM="$2"

cd Dispersed_${frb}/normalise_${DM}

cd channel_wise
for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp $REDIS/scripts/cudatemp.sh ${scriptname}

    echo "cd $REDIS/normalise_$DM/channel_wise" >> ${scriptname}
    echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
    echo $runcudafdmt ${f} -t 1024 -d 8192 -o ${f}.cand >> ${scriptname}
    echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
    echo python $REDIS/fredda/fredfof.py ${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done

cd ../whole_grid
for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp ../../../scripts/cudatemp.sh ${scriptname}

    echo "cd $REDIS/normalise_$DM/whole_grid" >> ${scriptname}
    echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
    echo $runcudafdmt ${f} -t 1024 -d 8192 -o ${f}.cand >> ${scriptname}
    echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
    echo python $REDIS/fredda/fredfof.py ${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done
