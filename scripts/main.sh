#!/bin/bash
#!/home/chip/anaconda3/envs/py37/bin/python

echo ""
echo "----------------------------------------"
echo "Training network"
echo "----------------------------------------"
echo ""
python ../train.py --net $1 --dataset $2 --trial $3 --epochs $4

echo ""
echo "----------------------------------------"
echo "Building graph"
echo "----------------------------------------"
echo ""
python ../build_graph_functional.py --net $1 --dataset $2 --trial $3 --epochs $5

echo ""
echo "----------------------------------------"
echo "Computing topology"
echo "----------------------------------------"
echo ""

./topology.sh $1 $2 $3

echo ""
echo "----------------------------------------"
echo "Prepare topology results"
echo "----------------------------------------"
echo ""

THRESHOLDS="0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55"
python prepare_results.py --net $1 --dataset $2 --trial $3 --epochs $5  --thresholds $THRESHOLDS
