#!/bin/bash
# source /fred/oz002/jhoffmann/RedisperseFRB/setup.sh > /dev/null
cd $REDIS

for f in "$@"
do

    while read -r line
    do
        if [ "${line:0:1}" == "#" ]
        then
            echo "$line, mjd" > temp.txt
        else
            echo "$line 0" >> temp.txt
        fi
    done < "$f"
    mv temp.txt $f
    
done

