#!/bin/bash

rm -f incSM.log

make

# Run the program with different parameters
for coremax in {1..81}
do
    ./vectorAdd -threads 1024 -coremax $coremax >> incSM.log
done

python3 plot.py