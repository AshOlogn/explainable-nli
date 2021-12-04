#!/bin/bash
MODEL=bart-expl
DATASET=anli-1
LR=1e-5
ALPHA=0.5
NUM_EPOCH=15
LOAD_PATH=trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/model_epoch-2_steps-190000_acc-90.6.pt

LOG_NAME="${MODEL}_${DATASET}_${LR}"
PROJ_DIR=$(dirname $BASH_SOURCE)/..
OUTPUT_DIR=out

cd $PROJ_DIR
mkdir -p $OUTPUT_DIR

sbatch --job-name=$LOG_NAME \
    --output=$OUTPUT_DIR/$LOG_NAME.o%j \
    --error=$OUTPUT_DIR/$LOG_NAME.e%j \
    --export=MODEL=$MODEL,DATASET=$DATASET,LR=$LR,ALPHA=$ALPHA,NUM_EPOCH=$NUM_EPOCH,LOAD_PATH=$LOAD_PATH \
    scripts/jobscript.sbatch
