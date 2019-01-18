#!/bin/bash

PATH="/data/data1/datasets/cvpr2019/adjacency/"

for e in $4
do
    eval "../cpp/symmetric "$PATH$1"_"$2"/badj_epc"$e"_t"$5"_trl"$3".csv 1 0"	
done
