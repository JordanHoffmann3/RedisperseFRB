#!/bin/bash

# module load anaconda3/2021.05
# export PYTHONPATH=/fred/oz002/hqiu/CRAFT_simulation/simfred/

source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do 
    cd $REDIS

    # Make directories
    if ! [ -d Dispersed_${frb}/pulse_injection/ ]
    then
        mkdir Dispersed_${frb}/pulse_injection/
    fi

    f=Data/${frb}/${frb}_param.dat

    # Get FRB parameters from file frb_param.dat
    f_low=$(sed -n 3p $f | awk '{print $1;}')
    f_high=$(sed -n 4p $f | awk '{print $1;}')
    t_int=$(sed -n 5p $f | awk '{print $1;}')
    w=$(sed -n 6p $f | awk '{print $1;}')
    sig=$(bc <<< "scale=1; $w/2.0")
    # A=$(sed -n 8p $f | awk '{print $1;}')

    # f_round=$(printf "%.0f\n" "$f_high")

    if [ $sig == "0" ]
    then
        sig="0.1"
    fi

    cp scripts/pulse_injection_temp.slurm Dispersed_$frb/pulse_injection/job.slurm
    echo cd $REDIS/Dispersed_$frb/pulse_injection/
    echo python3 ../../src/pulse_injection.py -l $frb -A 100 -N 20 --dm_start 0 --dm 3000 --sig_start $sig --sig $sig --bwchan -1 --fch1 $f_high --tsamp $t_int >> Dispersed_$frb/pulse_injection/job.slurm
    cd Dispersed_$frb/pulse_injection
    sbatch job.slurm

    echo "frb: ${frb} job scheduled"

done
