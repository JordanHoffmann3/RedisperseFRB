source /fred/oz002/psrhome/scripts/psrhome_2019-03.sh
module load cuda/9.0.176
module load gcc/6.4.0
module load anaconda2/5.1.0
export CUDA=$CUDA_PATH
export CRAFT=/fred/oz002/hqiu/CRAFT_simulation/craftrepo/craft/
export PATH=/fred/oz002/hqiu/CRAFT_simulation/craftrepo/craft/python/:{$PATH}
alias runcudafdmt=$CRAFT/cuda-fdmt/cudafdmt/src/cudafdmt
export PYTHONPATH=/fred/oz002/hqiu/CRAFT_simulation/craftrepo/craft/python/:{$PYTHONPATH}
