#!/bin/bash
#SBATCH -J train_expl_alpha-0.2_lr-2e-5
#SBATCH -o out/train_expl_alpha-0.2_lr-2e-5.o%j
#SBATCH -e out/train_expl_alpha-0.2_lr-2e-5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 08:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=esnli
export BATCH_SIZE=8
export LR=2e-5

python -u models/train_expl_model.py \
--task=train \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda \
--alpha=0.8 \
--learning_rate=$LR \
--num_train_epochs=3 \
--validation_steps=250
