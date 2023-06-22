#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

for frb in "$@"
do

    cd Dispersed_${frb}

    if ! [ -d cuda_job/ ]
    then
        mkdir cuda_job
        mkdir fredda_outputs
    fi
    
    cd outputs

    # For each fil file
    for f in *.fil
    do

        if [ ! -f ../fredda_outputs/${f}.cand.fof ]
        then
            cd ../cuda_job

            # Create and run cudascript
            scriptname=cudarun_${f}.sh
            cp $REDIS/scripts/cudatemp.sh ${scriptname}

            echo "cd $REDIS/Dispersed_${frb}" >> ${scriptname}
            echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
            echo $runcudafdmt outputs/${f} -t 1024 -d 8192 -x 9 -o fredda_outputs/${f}.cand >> ${scriptname}
            # echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
            echo python $REDIS/fredda/fredfof.py --dmmin=-1.0 fredda_outputs/${f}.cand >> ${scriptname}

            chmod 755 ${scriptname}

            sbatch ${scriptname}

            cd ../outputs
        fi
    done

    cd ..
done

