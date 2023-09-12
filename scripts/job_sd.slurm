#!/bin/bash
#SBATCH --job-name=REDISPERSION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --mem=150G

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
source $REDIS/loadpy.sh

#For each FRB
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

    DM=${DMi}

    # Make directories
    if ! [ -d Outputs/Dispersed_${frb}/sd_${DM}/ ]
    then
        mkdir Outputs/Dispersed_${frb}/sd_${DM}
        mkdir Outputs/Dispersed_${frb}/sd_${DM}/channel_wise
        mkdir Outputs/Dispersed_${frb}/sd_${DM}/whole_grid
    fi

    # Remove existing DM 0 file in case it was produced with different flags
    if [ -f Outputs/Dispersed_${frb}/sd_${DM}/channel_wise/${frb}_DM_0.0.npy ]
    then
        rm Outputs/Dispersed_${frb}/sd_${DM}/channel_wise/${frb}_DM_0.0.npy
    fi
    if [ -f Outputs/Dispersed_${frb}/sd_${DM}/whole_grid/${frb}_DM_0.0.npy ]
    then
        rm Outputs/Dispersed_${frb}/sd_${DM}/whole_grid/${frb}_DM_0.0.npy
    fi
    
    for i in 1 2 3 4 5 6 7 8 9 10 15 20 30
    do
        # Do redispersion
        python src/redisperse.py ${frb} ${DMi} ${DM} ${f_mid} ${t_int} --sd_f=${i} --out=Outputs/Dispersed_${frb}/sd_${DM}/whole_grid/
        mv Outputs/Dispersed_${frb}/sd_${DM}/whole_grid/${frb}_DM_${DM}.fil Outputs/Dispersed_${frb}/sd_${DM}/whole_grid/${frb}_DM_${DM}_${i}.fil        

        python src/redisperse.py -n 1 ${frb} ${DMi} ${DM} ${f_mid} ${t_int} --sd_f=${i} --out=Outputs/Dispersed_${frb}/sd_${DM}/channel_wise/
        mv Outputs/Dispersed_${frb}/sd_${DM}/channel_wise/${frb}_DM_${DM}.fil Outputs/Dispersed_${frb}/sd_${DM}/channel_wise/${frb}_DM_${DM}_${i}.fil
    done    

done

