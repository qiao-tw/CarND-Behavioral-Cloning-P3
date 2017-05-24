import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# skip 1st line
lines = lines[1:]

# generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # center
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # center_inverse
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle * -1.0)
                # left
                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = center_angle + 0.2
                images.append(left_image)
                angles.append(left_angle)
                # left_inverse
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle * -1.0)
                # right
                name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = center_angle - 0.2
                images.append(right_image)
                angles.append(right_angle)
                # center_inverse
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle * -1.0)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# create model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6, validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch=8)

model.save('model.h5')
exit()
