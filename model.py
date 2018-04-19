import sys
import csv
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, BatchNormalization
from keras.layers import Dropout
from keras.models import load_model
from keras import regularizers
import keras
import math

def generator(samples, batch_size=64):
	num_samples = len(samples)
	while 1:
		shuffle(lines)
		for offset in range(0, num_samples, batch_size):
			end = offset + batch_size
			batch_samples = samples[offset:end]

			images = []
			angles = []
			for batch_sample in batch_samples:
				steering_center = float(batch_sample[3])
				correction = 0.5
				steering_left = steering_center + correction   # steer more to the righ
				steering_right = steering_center - correction  # steer more to the left
				for i in range(0, 3):
					source_path = batch_sample[i]
					filename = source_path.split("/")[-1]
					current_path = os.path.join(image_dir, "data/IMG/", filename)
					image = cv2.imread(current_path)
					images.append(image)
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)

			# Data Augmentation
			augmented_images, augmented_angles = [], []
			for image, angle in zip(images, angles):
		        	augmented_images.append(image)
		        	augmented_angles.append(angle)
		        	augmented_images.append(cv2.flip(image, 1))
		        	augmented_angles.append(angle*-1.0)


			X_train = np.array(augmented_images)
			y_train =np.array(augmented_angles)
			yield shuffle(X_train, y_train)


image_dir = "../CarND-Behavioral-Cloning-P3/"
lines = []
with open("../CarND-Behavioral-Cloning-P3/data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

if os.path.exists("model.h5") and False:
	print("Find existed model.")
	model = load_model("model.h5")
	print("Load model succeeded")
else:
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
	# Input: (160, 320, 3) Output:(65, 320, 3)
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	# Input: (65, 320, 3) Output: (65, 320, 6)
	model.add(Conv2D(6, kernel_size=(5, 5), activation="relu", padding="same"))
	# Input: (65, 320, 6) Output: (32, 160, 6)
	model.add(MaxPooling2D())
	# Input: (32, 160, 6) Output: (32, 160, 16)
	model.add(Conv2D(16, kernel_size=(5, 5), activation="relu", padding="same"))
	# Input: (32, 160, 16) Output: (16, 80, 16)
	model.add(MaxPooling2D())
	# Input: (16, 80, 16) Output: (16, 80, 32)
	model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
	# Input: (16, 80, 32) Output: (8, 40, 32)
	model.add(MaxPooling2D())
	# Input: (8, 40, 32) Output: (8, 40, 64)
	model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
	# Input: (8, 40, 64) Output: (4, 20, 64)
	model.add(MaxPooling2D())
	# Input (4, 20, 64) Output: (5120)
	model.add(Flatten())
	# Input: 5120 Output: 1024
	model.add(Dense(1024, activation="relu"))
	model.add(Dropout(0.5))
	# Input: 1024 Output: 512
	model.add(Dense(512, activation="relu"))
	model.add(Dropout(0.5))
	# Input: 512 Output: 1
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/32), 
				     epochs=40,
		    validation_data=validation_generator,
		    validation_steps=math.ceil(len(validation_samples)/32))

model.save('model.h5')
print("Saved")

### print the keys contained in the history object
#print(history_object.history.keys())
### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
