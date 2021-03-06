from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import cv2
import numpy as np
import random
import os
import os.path
from IPython.display import clear_output
import clock
import math
import matplotlib.pyplot as plt
import pandas as pd

# Set Up GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

start_time = clock.now()        # Start Timer

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
NUM_CLASSES = len(CLASSES)              # The number of classes
IMG_SIZE = 28                          # Pixel-Width of images
BATCH_SIZE = 64	                    # The number of images to process during a single pass
EPOCHS = 20	                            # The number of times to iterate through the entire training set
IMG_ROWS, IMG_COLS = IMG_SIZE, IMG_SIZE # Input Image Dimensions
DATA_UTILIZATION = 1                    # Fraction of data which is utilized in training and testing
VALIDATION_SPLIT = 0.2
DATA_FOLDER = "final_dataset"
NAME = "withoutphdmodelv7"
TEST_RATIO = 0.1

# Define function to read images from folder and convert them to gray scale
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0) # Reads the images from folder
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converts images to gray scale
        if img is not None:
            images.append(img)
    return images

# Define function to load data set and return testing and training data
def loadData(test_ratio = TEST_RATIO):
    data = np.zeros((0,IMG_ROWS, IMG_COLS))
    labels = []

    ### Load data from folder titled after the character class name eg. ('Zero', 'One', 'Two'...) and label it with a
    ### corresponding integer value eg. (0, 1, 2...)
    for i, CLASS in enumerate(CLASSES):
        images = load_images_from_folder(DATA_FOLDER + '/' + CLASS)         # Load images from folder
        print(CLASS +" loaded")
        data = np.concatenate((data,images), axis=0)    # Add features (images) to data variable
        # print("concatenated")
        label = len(images) * [i]                       # Create a list of the feature labels the length of the number of images
        labels = np.concatenate((labels, label), axis=0)# Append the list of labels to the labels variable

    sort_data = list(zip(data,labels))                  # Zip the together the labels and features
    random.shuffle(sort_data)                           # Shuffle the labels and features together
    data, labels = zip(*sort_data)                      # Unzip the labels-features variable back into data and labels

    # Delete proportion of data equal to 1-DATA_UTILIZATION to speed up training and testing
    data = data[0:int(len(data)*DATA_UTILIZATION)]
    labels = labels[0:int(len(data)*DATA_UTILIZATION)]

    # Split data into test and train sets
    cutoff = int((1 - TEST_RATIO) * len(data))          # Determine the index at which to split the dataset into test and train
    x_train = data[0:cutoff]                            # Training features
    y_train = labels[0:cutoff]                          # Training labels
    x_test = data[cutoff:]                              # Testing features
    y_test = labels[cutoff:]                            # Testing labels
    np.save(NAME + '_test',x_test) # saves test.npy
    np.save(NAME + '_test_labels',y_test)
    np.save('train',x_train)
    np.save('train_labels',y_train)
    # save numpy array as .npy formats
    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

# Load data
(x_train, y_train), (x_test, y_test) = loadData(test_ratio = TEST_RATIO)

x_test = x_test.reshape(x_test.shape[0],IMG_ROWS, IMG_COLS,1)     # Reshape x_test where 1 = number of colors
x_train = x_train.reshape(x_train.shape[0],IMG_ROWS, IMG_COLS,1)  # Reshape x_test
x_test = x_test.astype('float32')       # Convert x_test to float 32
x_test /= 255                           # Scale feature values from 0-255 to values from 0-1
input_shape = (IMG_ROWS, IMG_COLS,1)
x_train = x_train.astype('float32')     # Convert x_train to float32
x_train /= 255                          # Scale feature values from 0-255 to values from 0-1

# convert class vectors to train = keras.utils.to_categorical(y_train,NUM_CLASSES = None) to binary class matrices
# Arguments: y: Class vector to be converted into a matrix (integers from 0 to num_classes)
#           num_classes: total number of classes
y_train = to_categorical(y_train, num_classes = None)
y_test = to_categorical(y_test, num_classes = None)