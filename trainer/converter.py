
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('anil_hasyv2.h5')
tflite_model = converter.convert()
open('anil_hasyv2.tflite', 'wb').write(tflite_model)