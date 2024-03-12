#!/bin/bash

#PBS -A EDU5
#PBS -q edu-1
#PBS -l elapstim_req=00:05:00

module load python/3.8 cuda/12.3.0

cd $PBS_O_WORKDIR

source /work/EDU5/$USER/hands-on/venv/bin/activate

python solution.py
