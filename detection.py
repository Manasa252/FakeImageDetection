import tensorflow as tf
from tensorflow.keras import layers, models
def build_detection_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
input_shape = (224, 224, 3)
detection_model = build_detection_model(input_shape)
detection_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
