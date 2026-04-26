#!/bin/bash
#$ -M jboskovi@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -l h_rt=96:00:00
#$ -N sionna_3m

module load conda
module load cuda/11.8
module load cudnn/8.9.3

eval "$(conda shell.bash hook)"
conda activate sionna_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DRJIT_LIBLLVM_PATH=$CONDA_PREFIX/lib/libLLVM-21.so

cd /users/jboskovi/sionna_project/sionna_osm_scene
python3 dataset_v3/dataset_gen_3m.py
