#!/bin/bash
#SBATCH -J predict-test_esnli
#SBATCH -o out/predict-test_esnli.o%j
#SBATCH -e out/predict-test_esnli.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 00:40:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=esnli
export BATCH_SIZE=4
export LR=1e-5

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart \
--load_path=trained_models/bart_base_esnli_epochs-3_bs-4_lr-1e-05/model_epoch-2_steps-249000_acc-90.9.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart-expl \
--load_path=trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/model_epoch-2_steps-190000_acc-90.6.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda


