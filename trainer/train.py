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

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
NUM_CLASSES = len(CLASSES)              # The number of classes
IMG_SIZE = 28                          # Pixel-Width of images
BATCH_SIZE = 128	                    # The number of images to process during a single pass
EPOCHS = 30	                            # The number of times to iterate through the entire training set
IMG_ROWS, IMG_COLS = IMG_SIZE, IMG_SIZE # Input Image Dimensions
DATA_UTILIZATION = 1                    # Fraction of data which is utilized in training and testing
VALIDATION_SPLIT = 0.2
DATA_FOLDER = "hackru"

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
def loadData():
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
    cutoff = int(len(data))          # Determine the index at which to split the dataset into test and train
    x_train = data[0:cutoff]                            # Training features
    y_train = labels[0:cutoff]                          # Training labels
    np.save('train',x_train)
    np.save('train_labels',y_train)
    # save numpy array as .npy formats
    return (np.asarray(x_train), np.asarray(y_train))

# Load data
(x_train, y_train) = loadData()

x_train = x_train.reshape(x_train.shape[0],IMG_ROWS, IMG_COLS,1)  # Reshape x_test
input_shape = (IMG_ROWS, IMG_COLS,1)
x_train = x_train.astype('float32')     # Convert x_train to float32
x_train /= 255                          # Scale feature values from 0-255 to values from 0-1

# convert class vectors to train = keras.utils.to_categorical(y_train,NUM_CLASSES = None) to binary class matrices
# Arguments: y: Class vector to be converted into a matrix (integers from 0 to num_classes)
#           num_classes: total number of classes
y_train = to_categorical(y_train, num_classes = None)
print("x_train.shape = {}, y_train.shape = {}".format(x_train.shape, y_train.shape))
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')

x = Conv2D(32, kernel_size=(3, 3), strides=1)(inputs)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.1)(x)

x = Conv2D(64, kernel_size=(3, 3), strides=2)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.1)(x)

x = Conv2D(128, kernel_size=(3, 3), strides=2)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.1)(x)

x = Flatten()(x)
x = Dense(200)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.1)(x)

predications = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=predications)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
lr_decay = lambda epoch: 0.0001 + 0.02 * math.pow(1.0 / math.e, epoch / 3.0)
decay_callback = LearningRateScheduler(lr_decay, verbose=1)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    validation_split=VALIDATION_SPLIT, callbacks=[decay_callback])
model.save('anil.h5')
converter = tf.lite.TFLiteConverter.from_keras_model_file('anil.h5')
tflite_model = converter.convert()
open('anil.tflite', 'wb').write(tflite_model)

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for learning rate
plt.plot(history.history['lr'])
plt.title('model learning rate')
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')
plt.show()