#!/bin/bash
#SBATCH --cpus-per-task=4 # max CPU cores per GPU request (6 on Cedar, 16 on Graham).
#SBATCH --mem=16000M   # memory per node (in mbs)
#SBATCH --time=12:00:00 # time (DD-HH:MM) OR (hh:mm:ss) format
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source tensorflow/bin/activate
python ./rename_large_split_samples.py # script to run (change to Myelin UNet)
  
