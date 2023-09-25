import tensorflow as tf
from tensorflow.keras import layers, models
def build_detection_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
