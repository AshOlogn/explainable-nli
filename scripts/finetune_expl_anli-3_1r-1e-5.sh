#!/bin/bash
#SBATCH -J finetune_expl_anli-3_lr-1e-5
#SBATCH -o out/finetune_expl_anli-3_lr-1e-5.o%j
#SBATCH -e out/finetune_expl_anli-3_lr-1e-5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 12:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=anli-3
export BATCH_SIZE=8
export LR=1e-5

python -u models/train_model.py \
--task=train \
--model=bart-expl \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--save_model \
--overwrite_old_model_dir \
--load_path=trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/model_epoch-2_steps-190000_acc-90.6.pt \
--device=cuda \
--alpha=0.5 \
--learning_rate=$LR \
--num_train_epochs=10 \
--validation_steps=250

