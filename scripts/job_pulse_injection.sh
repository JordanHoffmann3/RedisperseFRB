#!/bin/bash

# module load anaconda3/2021.05
# export PYTHONPATH=/fred/oz002/hqiu/CRAFT_simulation/simfred/

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

for frb in "$@"
do 
    cd $REDIS

    # Make directories
    if ! [ -d Dispersed_${frb}/pulse_injection/ ]
    then
        mkdir Dispersed_${frb}/pulse_injection/
        mkdir Dispersed_${frb}/pulse_injection/job/
        mkdir Dispersed_${frb}/pulse_injection/outputs/
    fi

    f=Data/${frb}/${frb}_param.dat

    # Get FRB parameters from file frb_param.dat
    f_low=$(sed -n 3p $f | awk '{print $1;}')
    f_high=$(sed -n 4p $f | awk '{print $1;}')
    t_int=$(sed -n 5p $f | awk '{print $1;}')
    w=$(sed -n 6p $f | awk '{print $1;}')

    sig=$(bc <<< "scale=2; $w/2.0")

    f2=Dispersed_$frb/fredda_outputs/extracted_outputs.txt

    # Determine pulse brightness
    if [ -f $f2 ]
    then
        SNR=$(sed -n 2p $f2 | awk '{print $2;}')
        A=$(bc <<< "scale=1; $SNR*sqrt(sqrt($t_int^2 + $w^2))")
    else
        echo "Using fluence of 100"
        A=100
    fi

    if [ $sig == "0" ]
    then
        sig="0.01"
    fi

    cd Dispersed_${frb}/pulse_injection/job/

    cp $REDIS/scripts/pulse_injection_temp.slurm job.slurm
    echo cd $REDIS/Dispersed_$frb/pulse_injection/outputs >> job.slurm
    echo python3 $REDIS/src/pulse_injection.py -l $frb -A $A -N 20 --dm_start 0 --dm 3000 --sig_start $sig --sig $sig --bwchan -1 --fch1 $f_high --tsamp $t_int >> job.slurm
    
    sbatch job.slurm

    echo "frb: ${frb} job scheduled"

done
