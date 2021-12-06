#!/bin/bash
#SBATCH -J predict-dev_bart-expl_esnli_lr-1e-5
#SBATCH -o out/predict-dev_bart-expl_esnli_lr-1e-5.o%j
#SBATCH -e out/predict-dev_bart-expl_esnli_lr-1e-5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 5:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=esnli
export BATCH_SIZE=4
export LR=1e-5

python -u models/train_model.py \
--task=predict \
--predict_split=dev \
--model=bart-expl \
--load_path=trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/model_epoch-2_steps-190000_acc-90.6.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

