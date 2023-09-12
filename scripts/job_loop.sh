#!/bin/bash

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do
    cd $REDIS

    # Make directories
    if ! [ -d Outputs/Dispersed_${frb}/ ]
    then
        mkdir Outputs/Dispersed_${frb}
        mkdir Outputs/Dispersed_${frb}/job
        mkdir Outputs/Dispersed_${frb}/outputs
    fi

    f=Data/${frb}/${frb}_param.dat

    # Get FRB parameters from file frb_param.dat
    DMi=$(sed -n 2p $f | awk '{print $1;}')
    f_mid=$(sed -n 3p $f | awk '{print $1;}')
    t_int=$(sed -n 4p $f | awk '{print $1;}')

    echo "frb: ${frb} DMi: ${DMi}"

    # # Remove existing DM 0 file in case it was produced with different flags
    # if [ -f Dispersed_${frb}/outputs/${frb}_DM_0.0.npy ]
    # then
    #     rm Dispersed_${frb}/outputs/${frb}_DM_0.0.npy
    # fi

    # Navigate to directories where jobs are created and executed
    cd Outputs/Dispersed_${frb}/job

    # For each DM
    for DMint in {0..7500..50} #$DMi
    do
        # Make the DM a double with one decimal if it is an integer
        if [[ ${DMint} == *.* ]]
        then
            DM=${DMint}
        else
            DM=$(printf "%.1f" ${DMint})
        fi

        if [ ! -f ../outputs/${frb}_DM_${DM}.fil ]
        then
            cp $REDIS/scripts/job_temp.slurm job_${DM}.slurm
            echo python src/redisperse.py ${frb} ${DMi} ${DM} ${f_mid} ${t_int} >> job_${DM}.slurm

            sbatch job_${DM}.slurm
        fi
    done
done


