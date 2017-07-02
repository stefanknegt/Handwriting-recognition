REQUIREMENTS:
Python 2.7      (standard on ubuntu, otherwise https://www.python.org/download/releases/2.7/

PYTHON LIBS:
sys, os, pickle         (standard with python install)
numpy, scipy, skimage   (included in ANACONDA, https://www.continuum.io/downloads, or install with pip install)
matplotlib.pyplot       (needed for plotting, included in ANACONDA)
yaml, h5py              (pip install pyyaml h5py)
Tensorflow OR theano    (CNN library, both can be used, TENSORFLOW recommended, https://www.tensorflow.org/install/)
keras                   (sudo pip install keras, see https://keras.io/ for detailed guide on installing CUDA and CUDNN as well)

RUN INSTRUCTIONS
make sure the following files are in the same folder (unzip to folder):
preprocessing.py
names_red.txt
red_extended_cnn1_nodropout.h5
DSTR_system.py

first, make DSTR_system.py executable with:
chmod +x DSTR_system.py

Then you can run the system with:
./DSTR_system.py folder_with_pgm_images

This will automatically create all .xml files with the same names as the corresponding images in the same folder

python DSTR_system.py folder_with_pgm_images
works the same way. For easy use it is recommended to put this folder with images in the same folder as the 4 files needed for the system to work are in.