#!/bin/bash
#!/home/chip/anaconda3/envs/py37/bin/python

function usage {
    echo "usage: $main [-net network_name] [-dt dataset_name] [-trl trial] [-netr n_epochs_train] [-ete epochs_test] [-plb permute_labels] [-dtsub datasubset]"
    echo "  -n  Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)"
    echo "  -d   Specify dataset (e.g. mnist, cifar10, imagenet)"
    echo "  -t    Specify trial number. Used to differentiate btw multiple trainings of same setup."
    echo "  -e   Specify number of training epochs. "
    echo "  -g    Specify list of epochs for which graph building is going to be performed. Sequence of positive integers delimited by blank space."
    echo "  -p    Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels."
    echo "  -s  Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportio of data randomly sampled. "
   
    exit 1
}


while getopts n:d:t:e:g:p:s: option
do
    case "${option}"
    in
	n) NET=${OPTARG};;
	d) DATASET=${OPTARG};;
	t) TRIAL=${OPTARG};;
	e) N_EPOCHS_TRAIN=$OPTARG;;
	g) EPOCHS_TEST=$OPTARG;;
	p) PERM_LABELS=$OPTARG;;
	s) DATA_SUBSET=$OPTARG;;
    esac
done

echo $EPOCHS_TEST

echo ""
echo "----------------------------------------"
echo "Training network"
echo "----------------------------------------"
echo ""
python ../train.py --net $NET --dataset $DATASET --trial $TRIAL --epochs $N_EPOCHS_TRAIN --permute_labels $PERM_LABELS --subset $DATA_SUBSET

echo ""
echo "----------------------------------------"
echo "Building graph"
echo "----------------------------------------"
echo ""
python ../build_graph_functional.py --net $NET --dataset $DATASET --trial $TRIAL --epochs $EPOCHS_TEST

echo ""
echo "----------------------------------------"
echo "Computing topology"
echo "----------------------------------------"
echo ""

for e in $EPOCHS_TEST
do
    ./topology.sh $NET $DATASET $TRIAL $e
done

echo ""
echo "----------------------------------------"
echo "Prepare topology results"
echo "----------------------------------------"
echo ""

THRESHOLDS="0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55"
for e in $EPOCHS_TEST
do
    python prepare_results.py --net $NET --dataset $DATASET --trial $TRIAL --epochs $e  --thresholds $THRESHOLDS
done
