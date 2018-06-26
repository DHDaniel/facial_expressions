
#################

# This script processes all the images in the dataset, and puts them in an H5 file as matrices.
#

#################

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

LABELS_MAP = { 'happiness': 1, 'neutral': 0 }
H5_FILE = './dataset.hdf5'

def load_image(img_path):
    """
    Loads an image and turns it into a matrix.

    Arguments:
    img_path -- a string indicating the filepath of the image, either absolute or relative to the script.

    Returns:
    processed_img -- a NumPy array representing the image, of shape (128, 128, 1) [height, width, channels]
    """
    # read into a numpy array (from looking at the dataset, this should be [350,350])
    image = np.array(ndimage.imread(img_path, flatten=False))
    # resize the image to a (128, 128)
    image = scipy.misc.imresize(image, size=(128, 128))
    # show us the image being processed
    #plt.imshow(image, cmap=cm.gray)
    #plt.show()
    # reshape to (height, width, num_channels)
    processed_img = image.reshape((128, 128, 1))

    return processed_img


def create_image_batch(img_paths):
    """
    Creates a batch of size (num_examples, height=128, width=128, channels=1) and returns it.

    Arguments:
    img_paths -- list of image paths to load

    Returns:
    batch -- a (num_examples, height=128, width=128, channels=1) batch of the images in img_paths
    """
    size = len(img_paths)
    batch = np.zeros((size, 128, 128, 1))
    for index, path in enumerate(img_paths):
        batch[index, :, :, :] = load_image(path)

    return batch

def process_row_batch(row_batch):
    """
    Function that processes a CSV batch of rows from the legends.csv file.

    Arguments:
    csv_row -- a csv_row from the legends.csv file, containing the fields user.id, image, and emotion.

    Returns:
    num_examples -- the number of examples processed.
    img_paths -- a list of image path names.
    Y_labels -- a (num_examples, 1) vector containing whether the image shows a happy person (1) or a neutral person (0).
    """
    # We're using simple Python lists because we only need a label vector, nothing more.
    img_paths = []
    Y_labels = []
    classes = LABELS_MAP.keys()
    for row in row_batch:
        emotion = row['emotion']
        path = './images/{}'.format(str(row['image']))
        # if emotion is happy or neutral
        if emotion in classes:
            img_paths.append(path)
            Y_labels.append(LABELS_MAP[emotion])

    num_examples = len(Y_labels)
    Y_labels = np.array(Y_labels).reshape((num_examples, 1))

    return num_examples, img_paths, Y_labels

def shuffle_and_split(X, Y, ratio=[0.6, 0.2, 0.2]):
    """
    Shuffles the X and Y dataset pairs, and returns a training set, test set, and validation set.

    Arguments:
    X -- numpy array of shape (num_examples, height, width, channels) containing images.
    Y -- numpy array of shape (num_examples, 1) containing labels (1 or 0).
    ratio -- optional. A tuple with three items, containing the ratios to split the data for training, validation, and testing. For example, [0.6, 0.2, 0.2] will create the split 60% training, 20% validation and 20% test.

    Returns:
    train -- Python dict with X and Y keys containing training data.
    validation -- Python dict with X and Y keys for validation.
    test -- Python dict with X and Y keys for testing.
    """
    num_examples = X.shape[0]

    # shuffle around, keeping the same X, Y pairs
    permutation = list(np.random.permutation(num_examples))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    # calculate splits for data
    first_split = int(num_examples * ratio[0])
    second_split = first_split + int(num_examples * ratio[1])
    # dicts to return
    train, validation, test = {}, {}, {}
    # split data into their respective sets
    train['X'] = shuffled_X[:first_split, :, :, :]
    train['Y'] = shuffled_Y[:first_split, :]
    validation['X'] = shuffled_X[first_split:second_split, :, :, :]
    validation['Y'] = shuffled_Y[first_split:second_split, :]
    test['X'] = shuffled_X[second_split:, :, :, :]
    test['Y'] = shuffled_Y[second_split:, :]

    return train, validation, test

def create_dataset_pairs(path, file):
    """
    Helper function that creates an X, Y dataset pair at path :path:.
    Returns a reference to the X dataset and the Y dataset.
    The X dataset has shape (0, 128, 128, 1) originally (with variable maxshape). The same goes for Y.
    """
    x_path = path + '/X'
    y_path = path + '/Y'
    X = file.create_dataset(x_path, (0, 128, 128, 1), maxshape=(None, 128, 128, 1), dtype=np.float32)
    Y = file.create_dataset(y_path, (0, 1), maxshape=(None, 1), dtype=np.int8)
    return X, Y

def write_and_resize(dset, data):
    """
    Writes the data to the datasets and resizes them accordingly.
    """
    # resize the dataset to be previous size plus the first axis of the data (which is always the number of examples)
    dset.resize(dset.shape[0] + data.shape[0], axis=0)
    # append the data to the end. Index means to set all the last data.shape[0] places to the new data.
    dset[-data.shape[0]:] = data

def write_to_file(dsets, data):
    """
    Writes to file all the train, val, and test batches.

    Arguments:
    dsets -- Python dictionary containing all the different datasets (training_X, training_Y, ...)
    data -- Python dictionary containing 'train', 'validation' and 'test', each with a respective 'X' and 'Y'
    """
    # loop through names of datasets, which will always end in
    for name in dsets.keys():
        # this will be either X or Y
        type = name[-1]
        # this will be 'training', 'validation' or 'test'
        clss = name[:-2]
        # write to dataset
        write_and_resize(dsets[name], data[clss][type])

def add_to_datasets(rows, dsets):
    """
    Adds row information to datasets.

    Arguments:
    rows -- list of csv rows that have been batched for processing.
    dsets -- dictionary containing the different datasets (training_X, training_Y, ...)
    """
    num_processed, img_paths, labels = process_row_batch(rows)
    # return early if no images to process
    if len(img_paths) == 0:
        return

    X_batch = create_image_batch(img_paths)
    data = {}
    data['training'], data['validation'], data['test'] = shuffle_and_split(X_batch, labels)
    write_to_file(dsets, data)


def main():
    """
    Main program that goes through all the images and creates an H5 file with X_train, Y_train, X_test, and Y_test datasets.

    Note that this program WILL DELETE the existing H5_FILE in order to write it anew. Do not run this program if you do not wish to delete the file.
    """
    # file that contains the image names along with their labels
    image_mapping_csv = open('./data/legend.csv', 'r')
    reader = csv.DictReader(image_mapping_csv)
    # create H5 datasets to write to and gradually resize.
    try:
        os.remove(H5_FILE)
        print "{} deleted to begin overwrite.".format(H5_FILE)
    except Exception as e:
        print "File did not exists. Creating new file..."
    file = h5py.File(H5_FILE, "a")
    dsets = {}
    # create training datasets
    dsets['training_X'], dsets['training_Y'] = create_dataset_pairs('training', file)
    # create validation datasets
    dsets['validation_X'], dsets['validation_Y'] = create_dataset_pairs('validation', file)
    # create test datasets
    dsets['test_X'], dsets['test_Y'] = create_dataset_pairs('test', file)
    # current collection of rows. We do this to not constantly write to the H5 file.
    rows = []
    # the size of the batches to write. This indicates how often to write to the file.
    write_batch_size = 500
    # keeping count of how many rows we've processed
    count = 0
    for row in reader:
        rows.append(row)
        count += 1
        # writing to file every time batch_size is met
        if count % write_batch_size == 0:
            print "{} images processed...".format(count)
            add_to_datasets(rows, dsets)
            rows = []

    # adding leftover rows that did not form a full batch
    add_to_datasets(rows, dsets)
    file.close()



if __name__ == '__main__':

    main()
