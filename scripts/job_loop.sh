#!/bin/bash

source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do
    cd $REDIS

    # Make directories
    if ! [ -d Dispersed_${frb}/ ]
    then
        mkdir Dispersed_${frb}
        mkdir Dispersed_${frb}/job
        mkdir Dispersed_${frb}/outputs
    fi

    f=Data/${frb}/${frb}_param.dat

    # Get FRB parameters from file frb_param.dat
    DMi=$(sed -n 2p $f | awk '{print $1;}')
    f_low=$(sed -n 3p $f | awk '{print $1;}')
    f_high=$(sed -n 4p $f | awk '{print $1;}')
    t_int=$(sed -n 5p $f | awk '{print $1;}')

    echo "frb: ${frb} DMi: ${DMi}"

    # # Remove existing DM 0 file in case it was produced with different flags
    # if [ -f Dispersed_${frb}/outputs/${frb}_DM_0.0.npy ]
    # then
    #     rm Dispersed_${frb}/outputs/${frb}_DM_0.0.npy
    # fi

    # Navigate to directories where jobs are created and executed
    cd Dispersed_${frb}/job

    # For each DM
    for DMint in $DMi {0..4000..50}
    do
        if [[ ${DMint} == *.* ]]
        then
            DM=${DMint}
        else
            # Make the DM a double with one decimal
            DM=$(printf "%.1f" ${DMint})
        fi

        if [ ! -f ../outputs/${frb}_DM_${DM}.fil ]
        then
            cp ../../scripts/job_temp.slurm job_${DM}.slurm
            echo python src/redisperse.py ${frb} ${DMi} ${DM} ${f_low} ${f_high} ${t_int} >> job_${DM}.slurm

            sbatch job_${DM}.slurm
        fi
    done
done


