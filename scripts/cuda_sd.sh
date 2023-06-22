#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

frb="$1"
DM="$2"

cd Dispersed_${frb}/sd_${DM}

cd whole_grid
for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp ../../../scripts/cudatemp.sh ${scriptname}

    echo "cd $REDIS/Dispersed_${frb}/sd_${DM}/whole_grid" >> ${scriptname}
    echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
    echo $runcudafdmt ${f} -t 1024 -d 8192 -o ${f}.cand >> ${scriptname}
    echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
    echo python $REDIS/fredda/fredfof.py ${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done

cd ../channel_wise
for f in *.fil
do
    # Create and run cudascript
    scriptname=cudarun_${f}.sh
    cp ../../../cudatemp.sh ${scriptname}

    #echo "#SBATCH output=slurm_${f}.out" >> ${scriptname}
    echo "" >> ${scriptname}
    echo "cd /fred/oz002/jhoffmann/RedisperseFRB/Dispersed_${frb}/sd_${DM}/channel_wise" >> ${scriptname}
    echo "source ../../../loadmodules.sh > /dev/null" >> ${scriptname}
    echo ../../../cudafdmt ${f} -t 1024 -d 8192 -o ${f}.cand >> ${scriptname}
    echo python ../../../fredfof.py ${f}.cand >> ${scriptname}

    sbatch ${scriptname}
done

#cd ../column_wise
#for f in *.fil
#do
#    # Create and run cudascript
#    scriptname=cudarun_${f}.sh
#    cp ../../../cudatemp.sh ${scriptname}
#
#    #echo "#SBATCH output=slurm_${f}.out" >> ${scriptname}
#    echo "" >> ${scriptname}
#    echo "cd /fred/oz002/jhoffmann/RedisperseFRB/Dispersed_${frb}/sd_${DM}/column_wise" >> ${scriptname}
#    echo "source ../../../loadmodules.sh > /dev/null" >> ${scriptname}
#    echo ../../../cudafdmt ${f} -t 1024 -d 8192 -o ${f}.cand >> ${scriptname}
#    echo python ../../../fredfof.py ${f}.cand >> ${scriptname}
#
#    sbatch ${scriptname}
#done
