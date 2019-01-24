#!/bin/bash
#!/home/chip/anaconda3/envs/py37/bin/python

function usage {
    echo "usage: $main [-net network_name] [-dt dataset_name] [-trl trial] [-netr n_epochs_train] [-ete epochs_test] [-plb permute_labels] [-dtsub datasubset]"
    echo "  -n  Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)"
    echo "  -d  Specify dataset (e.g. mnist, cifar10, imagenet)"
    echo "  -t  Specify trial number. Used to differentiate btw multiple trainings of same setup."
    echo "  -l  Specify learnig rate for training. Recommendet: 0.001"
    echo "  -e  Specify number of training epochs. "
    echo "  -g  Specify list of epochs for which graph building is going to be performed. Sequence of positive integers delimited by blank space."
    echo "  -p  Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels."
    echo "  -s  Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportion of data randomly sampled. "
    exit 1
}

while getopts n:d:t:l:e:g:p:s: option
do
    case "${option}"
    in
	n) NET=${OPTARG};;
	d) DATASET=${OPTARG};;
	t) TRIAL=${OPTARG};;
	l) LR=${OPTARG};;
	e) N_EPOCHS_TRAIN=${OPTARG};;
	g) EPOCHS_TEST=${OPTARG};;
	p) PERM_LABELS=${OPTARG};;
	s) DATA_SUBSET=${OPTARG};;
    esac
done

source config.sh #define your static variables (SAVE_PATH, THRESHOLDS, LR, etc) in this file

echo ""
echo "----------------------------------------"
echo "Training network"
echo "----------------------------------------"
echo ""
python ../train.py --net $NET --dataset $DATASET --trial $TRIAL --epochs $N_EPOCHS_TRAIN --lr $LR --permute_labels $PERM_LABELS --subset $DATA_SUBSET 

echo ""
echo "----------------------------------------"
echo "Building graph"
echo "----------------------------------------"
echo ""
python ../build_graph_functional.py --net $NET --dataset $DATASET --trial $TRIAL --epochs $EPOCHS_TEST --thresholds $THRESHOLDS

echo ""
echo "----------------------------------------"
echo "Computing topology"
echo "----------------------------------------"
echo ""

for e in $EPOCHS_TEST
do
    for t in $THRESHOLDS
    do
	eval "../cpp/symmetric "$SAVE_PATH$NET"_"$DATASET"/badj_epc"$e"_t"$t"_trl"$TRIAL".csv 1 0"      
    done
done

echo ""
echo "----------------------------------------"
echo "Prepare topology results"
echo "----------------------------------------"
echo ""

python prepare_results.py --path $SAVE_PATH --net $NET --dataset $DATASET --trial $TRIAL --epochs $EPOCHS_TEST  --thresholds $THRESHOLDS --permute_labels $PERM_LABELS --subset $DATA_SUBSET
