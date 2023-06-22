#!/bin/bash
#
#SBATCH --job-name=process_fredda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=8g

# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS
source loadcuda.sh

#./cudafdmt hifreq_single_dm2300_width8.0.fil -t 4096 -d 8192
#./cudafdmt Dispersed_181112/181112_DM100.0.fil -t 4096 -d 8192
#./cudafdmt Dispersed_test/test_DM10.0.fil -t 4096 -d 8192
#./cudafdmt test.fil -t 4096 -d 8192
#fredfof.py fredda.cand
#cat fredda.cand.fof

level_flags="-P 1e9 -Q 9e10"
fredda_extra="-X $ICS_DADA_KEY"

$runcudafdmt -h #Dispersed_181112/outputs/181112_DM_589.265.fil -t 1024 -d 1024 -o fredda.cand
#./cudafdmt Dispersed_181112/181112_DM100.0.fil -t 4096 -d 8192 -n 16384 -r 0 -s 2 -T 0.25 -M 0.007 -K 1.0 -C 10 -z 4.5 $level_flags -F fredda_flags.txt -W 5 -x 3.0 -b 20 -o fredda.cand
#./cudafdmt hifreq_single_dm2300_width8.0.fil -t 4096 -d 8192 -n 16384 -r 1 -s 2 -T 0.25 -M 0.007 -K 1.0 -C 10 -z 4.5 $level_flags -F fredda_flags.txt -W 5 -x 7.5 -b 20 -o fredda.cand

# fredfof.py fredda.cand
# cat fredda.cand.fof

# Check -I = 1 gives same output as no -I with -G 0.25
# -F $icsdir/fredda_flags.txt
