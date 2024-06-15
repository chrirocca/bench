#!/bin/bash

SMnum=108
GPCnum=6
logFile="incGPC_${GPCnum}.log"

rm -f $logFile

make

# Run the program with different parameters
for gpc in $(seq 0 $GPCnum)
do
    gpc_list=$(seq -s, 0 $gpc)
    ./vectorAdd -threads 1024 -blocks $SMnum -gpc $gpc_list >> $logFile
done

python3 plot.py $GPCnum