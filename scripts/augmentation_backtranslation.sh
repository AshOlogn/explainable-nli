#!/bin/bash
#SBATCH -J backtranslation
#SBATCH -o out/backtranslation.o%j
#SBATCH -e out/backtranslation.e%j
#SBATCH -p gtx
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 6:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

python -u augmentation/translation.py
