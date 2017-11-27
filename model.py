import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, Dropout
from keras import regularizers
from keras.models import Sequential
from sklearn import utils
from sklearn import model_selection


def showImage(image, title="Image"):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def flip(image):
    return cv2.flip(image, 1)


def add_noise(image):
    noise = np.zeros_like(image)
    noise = cv2.randn(noise, (0, 0, 0), (255, 255, 255))
    noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)

    #showImage(noise, 'Generated noise')
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


def data_generator(driving_data_list, batch_size=2048):
    # organize images and measurements. fill arrays accordingly via a generator function
    steering_bias = 0.25  # tune this
    datapoints_count = len(driving_data_list)
    while 1:
        # using SKLearn to shuffle the pandas dataframe as suggested here:
        # https://stackoverflow.com/questions/15772009/shuffling-permutating-a-dataframe-in-pandas
        utils.shuffle(driving_data_list)
        driving_data_list = driving_data_list.reset_index(drop=True)

        for offset in np.arange(0, datapoints_count, batch_size):
            batch_data = driving_data_list[offset:offset+batch_size]
            batch_data = batch_data.reset_index(drop=True)
            print('Batch data len: {co}'.format(co=len(batch_data)))
            images = []
            steering_angles = []

            # as seen at: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            for row in batch_data.itertuples():
                # reading images in the following order:
                # center, left, right
                for column in np.arange(1, 4):
                    img = cv2.imread(row[column].strip(' '))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                steering_angles.append(float(row[4]))
                # add a corrective bias for each left image and -0.25 for each right camera image
                steering_angles.append(float(row[4]) + steering_bias)
                steering_angles.append(float(row[4]) - steering_bias)
            print("Loaded {num} images.".format(num=len(images)))
            print("Loaded {num} steering angles.".format(num=len(steering_angles)))

            # Flipping the images to extend the training set
            agu_images, agu_steering_angles = agument_dataset(images, steering_angles)

            # Test if agumented images are fine by picking a random image to be shown
            #showImage(agu_images[np.random.randint(0, len(agu_images))], "Agumented image")

            images.extend(agu_images)
            steering_angles.extend(agu_steering_angles)

            for i in np.arange(0, len(images)):
                images[i] = add_noise(images[i])
            print("Random noise added to each image.")

            print('Datapoints in batch: {count}'.format(count=len(steering_angles)))
            result = (len(steering_angles) == len(images))
            print('Steering angles count and image count match: {result}'.format(result=result))
            # Test if noise is added properly to the image
            #showImage(images[np.random.randint(0, len(images))], "Image with noise")
            yield utils.shuffle(np.array(images), np.array(steering_angles))


def create_keras_model():
    keras_model = Sequential()
    # Using NVIDIA's CNN Model as seen here:
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # Input plane (160 x 320 x 3)

    # Normalization layer
    # udacity lambda normalization layer as seen in the lectures
    keras_model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)))
    # Cropping the picture to include only relevant information
    keras_model.add(Cropping2D(((55, 25), (0,0))))

    # Add Gaussian noise to randomly agument the dataset.
    #keras_model.add(GaussianNoise(2))

    # Normalized input planes (160 x 320 x 3), cropped -> Conv2D, 5x5 kernel
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

    #driving_data = csv_import("data/driving_log_test_1.csv")
    #data_17 = csv_import("data/driving_log_test_2.csv")

    # Center, Left, Right, Steering, Throttle, Brake, Speed
    driving_data = driving_data.append(data_17, ignore_index=True)

    # Training and validation data split
    training_data, validation_data = model_selection.train_test_split(driving_data, test_size=0.3)

    training_data = training_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)

    # generate model and compile
    print('Compiling the KERAS model!')
    model = create_keras_model()
    model.compile(optimizer='adam', loss='mse')

    # train the model
    print('Training...')
    batch_size = 512
    model.fit_generator(
        data_generator(training_data, batch_size=batch_size),
        steps_per_epoch=2*len(training_data)/(6*batch_size), # Multiplied by 2 because of data agumentation.
                                                      # Three images per data point
        epochs=3,
        validation_data=data_generator(validation_data, batch_size=batch_size),
        validation_steps=2*len(validation_data)/(6*batch_size),
        shuffle=True
    )
    # save the model
    print('Saving model...')
    model.save('model.h5')
    print('Done!')
