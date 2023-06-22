#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

frb=$1

for date in "200101" "210101" "220101"
do

    ./scripts/revert_fredda.sh $date

    cd Dispersed_${frb}

    if ! [ -d fredda_old ]
    then
        mkdir fredda_old
    fi

    if ! [ -d fredda_old/fredda_outputs_$date ]
    then
        mkdir fredda_old/cuda_job_$date
        mkdir fredda_old/fredda_outputs_$date
    fi
    
    if ! [ -d reverse_fq ]
    then
        mkdir reverse_fq
        ../scripts/reverse_fils.sh $frb
    fi

    cd outputs

    # For each fil file
    for f in *.fil
    do

        if [ ! -f ../fredda_old/fredda_outputs_$date/${f}.cand.fof ]
        then
            cd ../fredda_old/cuda_job_$date

            # Create and run cudascript
            scriptname=cudarun_${f}.sh
            cp $REDIS/scripts/cudatemp.sh ${scriptname}

            echo "cd $REDIS/Dispersed_${frb}" >> ${scriptname}
            echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
            echo ${runcudafdmt}_${date} outputs/${f} -t 1024 -d 8192 -o fredda_old/fredda_outputs_$date/${f}.cand >> ${scriptname}
            # echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
            echo python $REDIS/fredda/fredfof.py fredda_old/fredda_outputs_$date/${f}.cand >> ${scriptname}

            chmod 755 ${scriptname}

            sbatch ${scriptname}

            cd ../../outputs
        fi
    done

    cd ../..
done

