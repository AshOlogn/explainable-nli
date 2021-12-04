#!/bin/bash
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 00:10:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH -A CS388
#SBATCH --mail-user=reid.li@utexas.edu
#SBATCH --mail-type=FAIL

source .venv/bin/activate
python -u models/train_model.py \
--task=train \
--model=$MODEL \
--dataset=$DATASET \
--batch_size=20 \
--device=cuda \
--alpha=$ALPHA \
--learning_rate=$LR \
--num_train_epochs=$NUM_EPOCH \
--validation_steps=1000 \
--load_path=$LOAD_PATH \
--error_analysis
