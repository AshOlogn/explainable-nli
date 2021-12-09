#!/bin/bash
#SBATCH -J train_large-expl_esnli_lr-1e-5_alpha-0.5
#SBATCH -o out/train_large-expl_esnli_lr-1e-5_alpha-0.5.o%j
#SBATCH -e out/train_large-expl_esnli_lr-1e-5_alpha-0.5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 10:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=esnli
export BATCH_SIZE=4
export LR=1e-5

python -u models/train_model.py \
--task=train \
--model=bart-expl \
--model_id=facebook/bart-large \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--save_model \
--overwrite_old_model_dir \
--device=cuda \
--alpha=0.5 \
--learning_rate=$LR \
--num_train_epochs=5 \
--validation_steps=1000

