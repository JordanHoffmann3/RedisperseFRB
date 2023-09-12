#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do

    if [ ${frb:0:6} -lt 190830 ]
    then
        echo "$frb : 190101"
        date=190101
    elif [ ${frb:0:6} -lt 200406 ]
    then
        echo "$frb : 200228"
        date=200228
    else
        echo "$frb : skipping"
        continue
    fi

    cd $REDIS/Outputs/Dispersed_${frb}

    if ! [ -d fredda_reverted ]
    then
        mkdir fredda_reverted
    fi

    if ! [ -d fredda_reverted/fredda_outputs_$date ]
    then
        mkdir fredda_reverted/cuda_job_$date
        mkdir fredda_reverted/fredda_outputs_$date
    fi
    
    if ! [ -d reverse_fq ]
    then
        mkdir reverse_fq
        ../scripts/reverse_fils.sh $frb
    fi

    $REDIS/scripts/revert_fredda.sh $date >> $REDIS/fredda/revert.log

    cd reverse_fq
    # cd outputs

    # For each fil file
    for f in *.fil
    do

        if [ ! -f ../fredda_reverted/fredda_outputs_$date/${f}.cand.fof ]
        then
            cd ../fredda_reverted/cuda_job_$date

            # Create and run cudascript
            scriptname=cudarun_${f}.sh
            cp $REDIS/scripts/cudatemp.sh ${scriptname}

            echo "cd $REDIS/Outputs/Dispersed_${frb}" >> ${scriptname}
            echo "source $REDIS/loadcuda.sh > /dev/null" >> ${scriptname}
            echo ${runcudafdmt}_${date} reverse_fq/${f} -t 1024 -d 8192 -o fredda_reverted/fredda_outputs_$date/${f}.cand >> ${scriptname}
            # echo "source $REDIS/loadpy.sh > /dev/null" >> ${scriptname}
            echo "if [ \$(awk '{print NF}' $REDIS/Outputs/Dispersed_$frb/fredda_reverted/fredda_outputs_$date/${f}.cand | tail -n 1) == '7' ]" >> ${scriptname}
            echo "then" >> ${scriptname}
            echo "  $REDIS/scripts/add_mjd.sh $REDIS/Outputs/Dispersed_$frb/fredda_reverted/fredda_outputs_$date/${f}.cand" >> ${scriptname}
            echo "fi" >> ${scriptname}
            echo "python $REDIS/fredda/fredfof.py fredda_reverted/fredda_outputs_$date/${f}.cand" >> ${scriptname}

            chmod 755 ${scriptname}

            sbatch ${scriptname}

            cd ../../reverse_fq
            # cd ../../outputs
        fi
    done

done

