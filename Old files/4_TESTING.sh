#!/bin/bash
#SBATCH --account=def-jantel
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=14
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source tensorflow/bin/activate
python ./4_Bergles_single_oligo_TESTING.py  # script to run (change to Myelin UNet)
