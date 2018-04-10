#!/bin/bash

### Parameters
outDir=experiments/Test5/
n_train=600
n_test=20
dataroot_images='data/train/images/'
dataroot_masks='data/train/mask/'

if [ $# -eq 1 ] ### if random arg directory is cleaned before optimization
then
    rm -r $outDir
    mkdir -p $outDir
    cp script.sh $outDir
else
    mkdir -p $outDir
fi
    
if [ $WHERE_AM_I == 'home' ]
then
    py='optirun python'
else #WHERE_AM_I='idiap'
    py='~/miniconda3/bin/python'
fi

echo 'Detec Nuclei ------------------------------------------------------'

image_size=160
n_crops=100
epochs=100

$py DetectNuclei.py --experiment $outDir --dataroot-images $dataroot_images\
    --dataroot-masks $dataroot_masks --cuda --batchSize 32 --epochs $epochs --n_crops $n_crops \
    --plots --lr 0.001 --imageSize $image_size --n_train $n_train --n_test $n_test
