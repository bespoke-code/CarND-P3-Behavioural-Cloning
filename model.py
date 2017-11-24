import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Sequential


def flip(image):
    return cv2.flip(image, 1)


def csv_import(csv_filepath):
    return pd.read_csv(csv_filepath)


def agument_dataset(images, measurements):
    # flip all images
    # invert all measurements to reflect the data

    assert (len(images) == len(measurements))
    agumented_dataset = []
    for image in images:
        agumented_dataset.append(flip(image))

    agumented_measurements = []
    for measurement in measurements:
        agumented_measurements.append(-1*measurement)

    return agumented_dataset, agumented_measurements


def create_keras_model() -> Sequential:
    model = Sequential()

    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))


    # Using NVIDIA's CNN Model as seen here:
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # Input plane (160 x 320 x 3)

    # Normalization layer
    # udacity lambda normalization layer as seen in the lectures
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)))

    # Add in the future and see how this should fit
    # model.add(Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)))

    # Normalized input planes (160 x 320 x 3) -> Conv2D, 5x5 kernel
    model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(Convolution2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


if __name__ == '__main__':
    # import csv tables
    log_name = "driving_log.csv"
    driving_data = csv_import("data/2016/" + log_name)
    data_17 = csv_import("data/2017/" + log_name)

    # Center, Left, Right, Steering, Throttle, Brake, Speed
    driving_data.append(data_17, ignore_index=True)

    # organize images and measurements. fill arrays accordingly
    # as seen at: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    images=[]
    for row in driving_data.itertuples():
        # reading images in the following order:
        # center, left, right
        for column in [0, 1, 2]:
            img = cv2.imread(row[column])
            # Maybe this will be useful:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    print("Loaded {num} images.".format(num=len(images)))

    steering_angles = []
    steering_bias = 0.2 # tune this
    for row in driving_data.itertuples():
        steering_angles.append( float(row[3]) )
        # add a corrective bias for each right image and -0.2 for each left camera image
        steering_angles.append( float(row[3]) + steering_bias)
        steering_angles.append( float(row[3]) - steering_bias)
    print("Loaded {num} steering angles.".format(num=len(steering_angles)))

    # generate model and compile
    model = create_keras_model()
    model.compile(optimizer='adam', loss='mse')

    # train the model
    model.fit(x=images, y=steering_angles, validation_split=0.3, shuffle=True)

    # save the model
    model.save('model.h5')