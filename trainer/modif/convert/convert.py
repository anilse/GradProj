import cv2
import os
import os.path

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
DATA_FOLDER = "hasyv2_test"
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0) # Reads the images from folder
        img = cv2.bitwise_not(img)
        #newimg = cv2.resize(img, (28, 28))
             # Converts images to gray scale
        cv2.imwrite(os.path.join(folder,filename),img)

for i, CLASS in enumerate(CLASSES):
    images = load_images_from_folder(DATA_FOLDER + '/' + CLASS)         # Load images from folder
