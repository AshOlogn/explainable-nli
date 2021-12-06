#!/bin/bash
#SBATCH -J train_expl-clf_esnli_lr-3e-5
#SBATCH -o out/train_expl-clf_esnli_lr-3e-5.o%j
#SBATCH -e out/train_expl-clf_esnli_lr-3e-5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 12:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=esnli
export BATCH_SIZE=16
export LR=1e-5

python -u models/train_model.py \
--task=train-expl-clf \
--dataset=$DATASET \
--save_model \
--overwrite_old_model_dir \
--batch_size=$BATCH_SIZE \
--device=cuda \
--learning_rate=$LR \
--num_train_epochs=10 \
--validation_steps=1000
