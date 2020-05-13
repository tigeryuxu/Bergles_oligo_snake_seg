#!/bin/bash
#SBATCH --cpus-per-task=3 # max CPU cores per GPU request (6 on Cedar, 16 on Graham).
#SBATCH --mem=24000M   # memory per node (in mbs)
#SBATCH --time=00:05:00 # time (DD-HH:MM) OR (hh:mm:ss) format
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source tensorflow/bin/activate
python ./print_figures_troubleshoot.py  # script to run (change to Myelin UNet)
