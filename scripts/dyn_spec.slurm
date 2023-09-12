#!/bin/bash
#SBATCH --job-name=make_dyn_spec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --mem=60G

frb=$1
f=Data/${frb}/${frb}_param.dat

# Get FRB parameters from file frb_param.dat
DMi=$(sed -n 2p $f | awk '{print $1;}')
f_mid=$(sed -n 3p $f | awk '{print $1;}')
t_int=$(sed -n 4p $f | awk '{print $1;}')

cd $REDIS

if [ -d Data/${frb}_cut/ ]
then
    rm -rf Data/${frb}_cut/
fi

mkdir Data/${frb}_cut/

cp Data/${frb}/${frb}_param.dat Data/${frb}_cut/${frb}_cut_param.dat
cp Data/${frb}/${frb}_X_t_${DMi}_full.npy Data/${frb}_cut/${frb}_cut_X_t_${DMi}_full.npy
cp Data/${frb}/${frb}_Y_t_${DMi}_full.npy Data/${frb}_cut/${frb}_cut_Y_t_${DMi}_full.npy

buf=$(bc <<< "scale=0; $2*100")
# python src/plot_HTR.py --dt=$2 --buf=$buf -e ${frb}_cut
python src/plot_HTR.py --dt=$2 --buf=20 -p 2.215 -e ${frb}_cut

if ! [ -d Outputs/Dispersed_${frb}_cut/ ]
then
    mkdir Outputs/Dispersed_${frb}_cut/
    mkdir Outputs/Dispersed_${frb}_cut/outputs
fi

if [ -f Outputs/Dispersed_${frb}_cut/outputs/${frb}_cut_DM_0.0.npy ]
then
    rm Outputs/Dispersed_${frb}_cut/outputs/${frb}_cut_DM_0.0.npy
fi

python src/redisperse.py -s ${frb}_cut $DMi 0.0 $f_mid $2
