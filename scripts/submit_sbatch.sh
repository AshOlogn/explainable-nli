#!/bin/bash
MODEL=bart-expl
DATASET=anli-1
LR=2e-5
ALPHA=.9
NUM_EPOCH=15

LOG_NAME="${MODEL}_${DATASET}_${LR}"
PROJ_DIR=$(dirname $BASH_SOURCE)/..
OUTPUT_DIR=out

cd $PROJ_DIR
mkdir -p $OUTPUT_DIR

sbatch --job-name=$LOG_NAME \
    --output=$OUTPUT_DIR/$LOG_NAME.o%j \
    --error=$OUTPUT_DIR/$LOG_NAME.e%j \
    --export=MODEL=$MODEL,DATASET=$DATASET,LR=$LR,ALPHA=$ALPHA,NUM_EPOCH=$NUM_EPOCH \
    scripts/jobscript.sbatch
