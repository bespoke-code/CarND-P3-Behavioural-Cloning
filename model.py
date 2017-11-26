import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Flatten, Dense, Lambda, GaussianNoise
from keras.layers import Convolution2D, Cropping2D, Dropout
from keras import regularizers
from keras.models import Sequential


def flip(image):
    return cv2.flip(image, 1)


def add_noise(image):
    noise = np.zeros_like(image)
    noise = cv2.randn(noise, (0, 0, 0), (255, 255, 255))
    noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.75, noise, 0.25, 0)


def csv_import(csv_filepath):
    return pd.read_csv(csv_filepath)


def agument_dataset(original_images, measurements):
    # flip all images
    # invert all measurements to reflect the data

    assert (len(original_images) == len(measurements))

    agumented_dataset = []
    for image in original_images:
        agumented_dataset.append(flip(image))

    agumented_measurements = []
    for measurement in measurements:
        agumented_measurements.append(-1*measurement)

    return agumented_dataset, agumented_measurements


def create_keras_model():
    keras_model = Sequential()

    keras_model.add(Flatten(input_shape=(160, 320, 3)))
    keras_model.add(Dense(1))

    # Using NVIDIA's CNN Model as seen here:
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # Input plane (160 x 320 x 3)

    # Normalization layer
    # udacity lambda normalization layer as seen in the lectures
    keras_model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)))

    # Add Gaussian noise to randomly agument the dataset.
    #keras_model.add(GaussianNoise(2))

    # Add in the future and see how this should fit
    # model.add(Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)))

    # Normalized input planes (160 x 320 x 3) -> Conv2D, 5x5 kernel
    keras_model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    keras_model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    keras_model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    keras_model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Flatten())
    keras_model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    keras_model.add(Dense(1))

    return keras_model


if __name__ == '__main__':
    # import csv tables
    log_name = "driving_log.csv"
    driving_data = csv_import("data/2016/" + log_name)
    data_17 = csv_import("data/2017/" + log_name)

    # Center, Left, Right, Steering, Throttle, Brake, Speed
    driving_data = driving_data.append(data_17, ignore_index=True)

    # organize images and measurements. fill arrays accordingly
    # as seen at: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    images = []
    for row in driving_data.itertuples():
        # reading images in the following order:
        # center, left, right
        for column in np.arange(1,4):
            img = cv2.imread(row[column].strip(' '))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    print("Loaded {num} images.".format(num=len(images)))

    steering_angles = []
    steering_bias = 0.25  # tune this
    for row in driving_data.itertuples():
        steering_angles.append(float(row[4]))
        # add a corrective bias for each left image and -0.25 for each right camera image
        steering_angles.append(float(row[4]) + steering_bias)
        steering_angles.append(float(row[4]) - steering_bias)
    print("Loaded {num} steering angles.".format(num=len(steering_angles)))

    # Flipping the images to extend the training set
    agu_images, agu_steering_angles = agument_dataset(images, steering_angles)
    images.extend(agu_images)
    steering_angles.extend(agu_steering_angles)

    for image in images:
        image = add_noise(image)
    print("Random noise added to each image.")


    # generate model and compile
    print("Compiling the KERAS model!")
    model = create_keras_model()
    model.compile(optimizer='adam', loss='mse')

    # train the model
    print("Training...")
    model.fit(x=np.array(images), y=np.array(steering_angles), validation_split=0.3, shuffle=True)

    # save the model
    print("Saving model...")
    model.save('model.h5')
    print("Done!")
