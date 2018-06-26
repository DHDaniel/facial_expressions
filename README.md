# Facial expressions

This repository is a fork of the original **muxspace/facial_expressions** repository. It adds a convenient script called `process_images.py` that processes the images in the `/images` folder and stores the information in an HDF5 file, ready to be used in machine learning algorithms.

# Setup

Clone this repository on your computer, and then run
```
$ python process_images.py
```
You must have Python 2.7 installed. Note that the script uses all of the following libraries, which you must have installed:
```python
import h5py
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import csv
import math
import os
```
Once you run the script, it will create a file called `dataset.hdf5` in the same directory as the script. It will have a size of **~800 MB**.

# Structure

The HDF5 file created will have the following structure:

- training
  - X
  - Y
- validation
  - X
  - Y
- test
  - X
  - Y

There are three groups, `training`, `validation`, and `test`, each with its corresponding `X` and `Y` datasets. `X` corresponds to the images, and `Y` to the labels. You can access each dataset using the `h5py` module in Python:
```python
import h5py
dset = h5py.File(filename, 'r')
training_X = dset['training/X']
training_Y = dset['training/Y']
```

# Shapes
Each `X` dataset is of shape **(batch_size, height, width, channels)**. In this case, we will be dealing only with 128x128 grayscale images, so the shapes for all the `X` datasets is `(batch_size, 128, 128, 1)`.

Each `Y` dataset is of shape **(batch_size, 1)**, where each entry is either `0` or `1`. A `0` corresponds to "neutral", indicating that the picture was of a neutral face, while a `1` corresponds to "happiness", indicating that the picture was of a face showing happiness (smiling).
