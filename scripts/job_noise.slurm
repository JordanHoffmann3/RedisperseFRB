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
    f_low=$(sed -n 3p $f | awk '{print $1;}')
    f_high=$(sed -n 4p $f | awk '{print $1;}')
    t_int=$(sed -n 5p $f | awk '{print $1;}')

    echo "frb: ${frb} DMi: ${DMi}"
    echo "${f_low}, ${f_high}, ${t_int}"

    DM=${DMi}

    # Make directories
    if ! [ -d Dispersed_${frb}/noise_${DM}/ ]
    then
        mkdir Dispersed_${frb}/noise_${DM}
    fi
    
    # Move existing filterbank temporarily
    if [ -f Dispersed_${frb}/outputs/${frb}_DM_${DM}.fil ]
    then
        mv Dispersed_${frb}/outputs/${frb}_DM_${DM}.fil Dispersed_${frb}/noise_${DM}/temp.fil
    fi

    # Remove existing DM 0 file in case it was produced with different flags
    if [ -f Dispersed_${frb}/outputs/${frb}_DM_0.0.npy ]
    then
        rm Dispersed_${frb}/outputs/${frb}_DM_0.0.npy
    fi

    # Run x times
    for i in {2..20}
    do
        # Do redispersion
        python src/redisperse.py -p ${frb} ${DMi} ${DM} ${f_low} ${f_high} ${t_int}
        
        mv Dispersed_${frb}/outputs/${frb}_DM_${DM}.fil Dispersed_${frb}/noise_${DM}/${frb}_DM_${DM}_${i}.fil
    done

    # Move original back
    if [ -f Dispersed_${frb}/noise_${DM}/temp.fil ]
    then
        mv Dispersed_${frb}/noise_${DM}/temp.fil Dispersed_${frb}/outputs/${frb}_DM_${DM}.fil
    fi
done

