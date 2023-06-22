#!/bin/bash
#SBATCH --job-name=single-run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --mem=150G

# module load python/3.8.5
# module load numpy/1.19.2-python-3.8.5
# module load matplotlib/3.3.2-python-3.8.5
# module load scipy/1.6.0-python-3.8.5
#./fred/oz002/hqiu/CRAFT_simulation/craftrepo/loadmodules.sh
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

cd $REDIS
source loadpy.sh
rm Dispersed_210912/outputs/210912_DM_0.0.npy

#python redisperse.py -s --tpad=0.0 test 0.0 0.0001 50 100 0.001
#python redisperse.py -p -s 181112 589.265 589.265 1130 1466 0.864
python src/redisperse.py -s 210912 1233.69 1000.0 1103.5 1439.5 1.182
#python redisperse.py -p -s 210117 731.02 0.0 1104 1440 0.05
