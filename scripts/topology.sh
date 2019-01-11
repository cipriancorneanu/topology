#!/bin/bash

THRESHOLDS="0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55"
PATH="/data/data1/datasets/cvpr2019/adjacency/"

for e in $5
do
    for t in $THRESHOLDS
    do
	eval "../cpp/symmetric "$PATH$1"_"$2"/badj_epc"$e"_t"$t"_trl"$3".csv 1 0"	
    done
done
