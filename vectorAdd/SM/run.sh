#!/bin/bash

SMnum=108
logFile="incSM_${SMnum}.log"

rm -f $logFile

make

# Run the program with different parameters
for ((coremax=1; coremax<SMnum+1; coremax=coremax+1))
do
    ./vectorAdd -threads 1024 -blocks $SMnum -coremax $coremax >> $logFile
done

python3 plot.py $SMnum