#!/bin/bash
#SBATCH --job-name=IncoherentDedispersion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --mem=16G

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
source $REDIS/loadpy.sh

for frb in "$@"
do
    cd $REDIS

    f=Data/${frb}/${frb}_param.dat

    # Get FRB parameters from file frb_param.dat
    DMi=$(sed -n 2p $f | awk '{print $1;}')
    f_low=$(sed -n 3p $f | awk '{print $1;}')
    f_high=$(sed -n 4p $f | awk '{print $1;}')
    t_int=$(sed -n 5p $f | awk '{print $1;}')

    echo "frb: ${frb} DMi: ${DMi} f_low: ${f_low} f_high: ${f_high}"

    # Make directories
    if ! [ -d Dispersed_${frb}/incoherent_dedispersion/ ]
    then
        mkdir Dispersed_${frb}/incoherent_dedispersion
    fi

    # Navigate to directories where jobs are created and executed
    cd Dispersed_${frb}/incoherent_dedispersion

    # Create the file
    echo "" > ${frb}_incoherent_SNR.txt

    # For each DM

    # for DMint in 50 #{25..3000..25}
    # do

    #     # Make the DM a double with one decimal
    #     DM=$(printf "%.1f" ${DMint})

    cd ../outputs
    for f in *.fil
    do
        cd ../incoherent_dedispersion
        DM="$(echo $f | awk -F'[_.]' '{print $3}').$(echo $f | awk -F'[_.]' '{print $4}')"

        if [ -f ../outputs/${frb}_DM_${DM}.fil ]
        then
            python $REDIS/src/incoherent_dedispersion.py -p -s 4 -n 15 ${frb} ${DM} ${f_low} ${f_high} ${t_int}
        else
            echo Error: No filterbank file for DM ${DM}
        fi
    done

    sort -k 1 -n $(printf "%s_incoherent_SNR.txt" ${frb}) > temp.txt
    mv temp.txt $(printf "%s_incoherent_SNR.txt" ${frb})
done


