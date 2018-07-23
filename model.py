import os
import csv

#Read the training data from two folders: /data for first lap, /Data_Video_3 for 2 and 3 lap with 
#recovering from the left and right sides of the road.
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('./Data_Video_3/driving_log.csv')as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Data loading is done')

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

#Define the generator
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #Because the file path in two folders are different, a if-else is needed.
                if len(batch_sample[0].split('/')) == 2:
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                else:
                    name =batch_sample[0]
                originalImage = cv2.imread(name)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                measurement = float(line[3])
                angles.append(measurement)
                
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*(-1.0))

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

ch, row, col = 3, 160, 320  # Trimmed image format
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D , Cropping2D
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=( row, col,ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
#Nvidia Net 
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1)
model.save('modeltry.h5')

print('The model is saved as modeltry.h5 .')

