from tensorflow.contrib import keras
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
import numpy as np
NAME = "withoutphdmodelv7"
x_test = np.load(NAME + "_test.npy")
y_test = np.load(NAME + "_test_labels.npy")
x_test = x_test.reshape(x_test.shape[0],28, 28,1)     # Reshape x_test where 1 = number of colors
x_test = x_test.astype('float32')       # Convert x_test to float 32
x_test /= 255                           # Scale feature values from 0-255 to values from 0-1
y_test = to_categorical(y_test, num_classes = None)
new_model = keras.models.load_model(NAME + '.h5')
new_model.summary()
score = new_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])