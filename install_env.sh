#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh 
conda create --name autograd python==3.10 -y
conda activate autograd
pip install numpy