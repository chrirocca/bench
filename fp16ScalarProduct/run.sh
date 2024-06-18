#!/bin/bash

rm -f incGPC.log

make

# Run the program with different parameters
for gpc in {0..5}
do
    gpc_list=$(seq -s, 0 $gpc)
    ./fp16ScalarProduct -gpc $gpc_list >> incGPC.log
done

python3 plot.py