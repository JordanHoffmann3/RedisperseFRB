#!/bin/bash
#SBATCH --job-name=REDISPERSION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --mem=60G

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
source $REDIS/loadpy.sh

# For each FRB
for frb in "$@"
do
    cd $REDIS

    f=Data/${frb}/${frb}_param.dat
    
    # Get FRB parameters from file frb_param.dat
    frb=$(sed -n 1p $f | awk '{print $1;}')
    DMi=$(sed -n 2p $f | awk '{print $1;}')
    f_mid=$(sed -n 3p $f | awk '{print $1;}')
    t_int=$(sed -n 4p $f | awk '{print $1;}')

    echo "frb: ${frb} DMi: ${DMi}"
    DM=0.0 #${DMi}

    # Make directories
    if ! [ -d Outputs/Dispersed_${frb}/offset_${DM}/ ]
    then
        mkdir Outputs/Dispersed_${frb}/offset_${DM}
    fi

    # Remove existing DM 0 file in case it was produced with different flags
    if [ -f Outputs/Dispersed_${frb}/offset_${DM}/${frb}_DM_0.0.npy ]
    then
        rm Outputs/Dispersed_${frb}/offset_${DM}/${frb}_DM_0.0.npy
    fi

    # Do for each offset
    for offset in {10..100..10}
    do
        # Do redispersion
        python src/redisperse.py ${frb} ${DMi} ${DM} ${f_mid} ${t_int} -o ${offset} --out=Outputs/Dispersed_${frb}/offset_${DM}/

        mv Outputs/Dispersed_${frb}/offset_${DM}/${frb}_DM_${DM}.fil Outputs/Dispersed_${frb}/offset_${DM}/${frb}_DM_${DM}_${offset}.fil
    done    

done

