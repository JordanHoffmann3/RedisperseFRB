#!/bin/bash
cd $REDIS

for frb in "$@"
do

    cd Dispersed_${frb}/pulse_injection

    # For each fil file
    for f in *.fil
    do

        if [ ! -f ../fredda_outputs/${f}.cand.fof ]
        then

            # Create and run cudascript
            scriptname=cudarun_${f}.sh
            cp $REDIS/scripts/cudatemp.sh ${scriptname}

            echo "cd $REDIS/Dispersed_${frb}/pulse_injection/" >> ${scriptname}
            echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
            echo $runcudafdmt ${f} -t 1024 -d 8192 -x 9 -o ${f}.cand >> ${scriptname}
            # echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
            echo python $REDIS/fredda/fredfof.py --dmmin=-1.0 ${f}.cand >> ${scriptname}

            chmod 755 ${scriptname}

            sbatch ${scriptname}

        fi
    done

    cd ../..
done

