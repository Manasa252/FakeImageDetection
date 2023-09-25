import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

real_images_dir = 'Images_data/fake'
fake_images_dir = 'Images_data/real'

real_images = []
fake_images = []

for filename in os.listdir(real_images_dir):
    img = Image.open(os.path.join(real_images_dir, filename))
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    real_images.append(img)

for filename in os.listdir(fake_images_dir):
    img = Image.open(os.path.join(fake_images_dir, filename))
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    fake_images.append(img)
real_labels = np.zeros(len(real_images))
fake_labels = np.ones(len(fake_images))
X = np.concatenate((real_images, fake_images), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
