#!/bin/bash
#SBATCH -J predict-test_anli
#SBATCH -o out/predict-test_anli.o%j
#SBATCH -e out/predict-test_anli.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 00:30:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export DATASET=anli-3
export BATCH_SIZE=4
export LR=1e-5

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart \
--load_path=trained_models/bart_anli-3_epochs-10_bs-4_lr-1e-05/model_epoch-7_steps-172750_acc-49.8.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart \
--load_path=trained_models/bart_finetune_anli-3_epochs-10_bs-8_lr-1e-05/model_epoch-7_steps-79000_acc-49.5.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart-expl \
--load_path=trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/model_epoch-9_steps-211750_acc-49.2.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart-expl \
--load_path=trained_models/bart-expl_finetune_anli-3_alpha-0.5_epochs-10_bs-8_lr-1e-05/model_epoch-7_steps-82000_acc-48.8.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

python -u models/train_model.py \
--task=predict \
--predict_split=test \
--model=bart-expl \
--load_path=trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/model_epoch-9_steps-330250_acc-50.8.pt \
--dataset=$DATASET \
--batch_size=$BATCH_SIZE \
--device=cuda

