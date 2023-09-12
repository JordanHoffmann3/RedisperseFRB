#!/bin/bash
for frb in "$@"
do
    cd $REDIS

    if ! [ -d Outputs/Dispersed_${frb}/pulse_injection/fredda_job/ ]
    then
        mkdir Outputs/Dispersed_${frb}/pulse_injection/fredda_job/
        mkdir Outputs/Dispersed_${frb}/pulse_injection/fredda_outputs/
    fi

    cd Outputs/Dispersed_${frb}/pulse_injection/outputs/

    # For each fil file
    for f in *.fil
    do

        if [ ! -f ../fredda_outputs/${f}.cand.fof ]
        then
            cd ../fredda_job

            # Create and run cudascript
            scriptname=cudarun_${f}.sh
            cp $REDIS/scripts/cudatemp.sh ${scriptname}

            echo "cd $REDIS/Outputs/Dispersed_${frb}/pulse_injection/fredda_outputs/" >> ${scriptname}
            echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
            echo $runcudafdmt ../outputs/${f} -t 1024 -d 8192 -x 9 -o ${f}.cand >> ${scriptname}
            # echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
            echo python $REDIS/fredda/fredfof.py --dmmin=-1.0 ${f}.cand >> ${scriptname}

            chmod 755 ${scriptname}

            sbatch ${scriptname}

        fi
    done

done

