## import all required libraries
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
## open file containing path to images and driving angles
with open ('training_data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        samples.append(line)

##keep 25% data for validation
train_samples, validation_samples = train_test_split(samples, test_size=0.25)

#correction angle for left and right camera images when they are seen by centre camera
correction = 0.3

##generator function as required by rubric
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'training_data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = 'training_data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = 'training_data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)

                center_angle = float(batch_sample[3])

                ##angle for left and right image when seen by centre camera
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            ## adding flipped images and -ve angles so neural network is not biased to this track
            augmented_images, augmented_angles= [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
        yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
print(train_generator)
validation_generator = generator(validation_samples, batch_size=32)

## Model same as Lenet with different input size to match image taken by car camera
model=Sequential()

## Normalizing images
##input_shape = 160,320,3 output_shape=160,320,3
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))

## Cropping images to remove unwanted area like sky, landscape above horizon and car hood
##output_shape = 85, 320, 3
model.add(Cropping2D(cropping=((70,25),(0,0))))

##output_shape = 81, 316, 6
model.add(Convolution2D(5,5,6, activation="relu"))

##output_shape = 40, 158, 6
model.add(MaxPooling2D())

##output_shape = 36, 154, 6
model.add(Convolution2D(5,5,6,activation="relu"))

#output_shape = 18, 77, 6
model.add(MaxPooling2D())

#output_shape = 8316
model.add(Flatten())
model.add(Dropout(0.5))

#output_shape = 120
model.add(Dense(120))
model.add(Dropout(0.5))

#output_shape = 84
model.add(Dense(84))
model.add(Dropout(0.5))

#output_shape = 1
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=4992, validation_data=validation_generator, nb_val_samples=1664, nb_epoch=4)

print(model.summary())
model.save('model_lenet.h5')
exit()
