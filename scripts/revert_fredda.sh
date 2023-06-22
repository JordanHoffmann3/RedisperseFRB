#!/bin/bash

# module load git/2.18.0
# module load gcc/6.4.0
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null

cd $CRAFT

frb=$1

if ! [ -f $REDIS/fredda/cudafdmt_$frb ]
then
    source $REDIS/loadcuda.sh
    
    date=$(printf "20%s-%s-%s" ${frb:0:2} ${frb:2:2} ${frb:4:2})

    commit_id=$(git log --until="${date}" | head -1 | awk '{print $2;}')

    echo "Checking out commit ${commit_id}"
    git checkout ${commit_id} .
    echo "Making..."
    cd cuda-fdmt/cudafdmt/src
    make
    echo "Done making"

    cp cudafdmt $REDIS/fredda/cudafdmt_$frb

    git checkout dadain
fi