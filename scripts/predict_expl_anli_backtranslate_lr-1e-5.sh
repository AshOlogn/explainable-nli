#!/bin/bash
#SBATCH -J predict-dev_bart-expl_anli_backtranslate_lr-1e-5
#SBATCH -o out/predict-dev_bart-expl_anli_backtranslate_lr-1e-5.o%j
#SBATCH -e out/predict-dev_bart-expl_anli_backtranslate_lr-1e-5.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 1:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=anli-3
export BATCH_SIZE=4
export LR=1e-5

python -u models/train_model.py \
--task=predict \
--predict_split=dev \
--model=bart-expl \
--load_path=trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/model_epoch-9_steps-330250_acc-50.8.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

