Compute canada:
(1) Create virtual environment: - this will become main folder you work in
module load python/3.6   ### might be at 3.7 at this point
virtualenv --no-download pytorch
source pytorch/bin/activate        ### this activates the virtual environment anytime you want to use it

(2) Installation of python packages (within virtual environment):
	pip install matplotlib scipy scikit-image pillow numpy natsort opencv-python pandas skan sklearn torchio tifffile


### For Graham builds:
pip install Cython
pip install --upgrade pip   ### to be able to install opencv-python



### other side notes:
	# ***installing sklearn requires Cython!
	# (SKAN ==> requires pandas, numba... ==> DOES NOT WORK ON BELUGA, numba is broken)
	# *** pip install tifffile NO LONGER WORKS??? (or does it?)

(3) Job submission script template:
#!/bin/bash
#SBATCH --gres=gpu:1   # requests GPU "generic resource"
#SBATCH --cpus-per-task=10 # max CPU cores per GPU request (6 on Cedar, 16 on Graham).
#SBATCH --mem=80000M   # memory per node (in mbs)
#SBATCH --time=02-00:00 # time (DD-HH:MM) OR (hh:mm:ss) format
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source pytorch/bin/activate
python ./filename.py


##############################################################################################################
If installing on home computer, you might want to get an IDE or something. I normally work through Spyder (which comes with Anaconda installation) for Python.
	- Anaconda is definitely great to have

##############################################################################################################

Graphics card drivers and compatability (may be tricky with windows)
# CUDA Toolkit ==> needs VIsual studio (base package sufficient???)
# CuDnn SDK ==> 
# Ignore step about putting cudnn with Visual Studio

###############################################################################################################

Install for Pytorch (***on compute canada AND home computer)
 (1) Go to website and get the custom install command that should look something like this: conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
          
     torch.__version__ to check install in Python
     
     ***For uninstall:
     conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  
     
***if gives error about pyqt5, need to
pip uninstall pyqt5
pip install pyqt5==5.12.0
then retry



(4) Location of training data on lab computer:
	- main data from old paper for training (already pre-processed to now work with training):
		D:\Tiger\Tiger\AI stuff\z_myelin_data_FULL_with_NANOFIBERS_TRAINING

	- raw data of the above and all raw ROIs generated from manually tracing by Tiger, Daryan, Matthew, and Qiao Ling:
		J:\DATA_2017-2018\Tiger_old_data\Daryan Images and ROIs


	- QL data for training on HUMAN OLs
		D:\Tiger\Tiger\AI stuff\z_myelin_data_1024x1024-QL-human-images


D:\Tiger\Tiger\MATLAB source\3) Super-resolution ==> for Tiger, ignore this file



