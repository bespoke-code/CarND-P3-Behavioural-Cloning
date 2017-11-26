import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Sequential
from os import getcwd

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
        agumented_measurements.append(-1 * measurement)

    return agumented_dataset, agumented_measurements


if __name__ == '__main__':
    driving_data = csv_import("data/driving_log_test_1.csv")
    dataset2 = csv_import("data/driving_log_test_2.csv")
    driving_data = driving_data.append(dataset2, ignore_index=True)
    #print(dataset1.iloc[3][:])
    #print(dataset2.loc[3])
    #print(driving_data.append(dataset2, ignore_index=True))
    print(driving_data.shape)
    images = []
    for row in driving_data.itertuples():
        # reading images in the following order:
        # center, left, right
        #print(row)
        for column in np.arange(1, 4):
            #print("Reading ", str(row[column]).strip(' '))
            img = cv2.imread(str(row[column]).strip(' '))
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    print("Loaded {num} images.".format(num=len(images)))
    #plt.imshow(images[int(np.random.random_integers(0, len(images), size=1))])
    #plt.show()

    steering_angles = []
    steering_bias = 0.25  # tune this
    for row in driving_data.itertuples():
        steering_angles.append(float(row[4]))
        # add a corrective bias for each left image and -0.2 for each right camera image
        steering_angles.append(float(row[4]) + steering_bias)
        steering_angles.append(float(row[4]) - steering_bias)
    print("Loaded {num} steering angles.".format(num=len(steering_angles)))

    agu_imgs, agu_steer = agument_dataset(images, steering_angles)
    #plt.imshow(agu_imgs[int(np.random.random_integers(0, len(agu_imgs), size=1))])
    #plt.show()
    #print(images)
    print(len(images))
    images.append(agu_imgs)
    #print(images)
    print(len(images))

    #print(steering_angles)
    #steering_angles.extend(agu_steer)
    #print(steering_angles)

    exit(0)
    # Test print
    #print(dataset.loc[3][6])

    image_name = "data/2016/IMG/left_2016_12_01_13_34_23_036.jpg"
    image = cv2.imread(image_name)

    noise = np.zeros_like(image)
    noise = cv2.randn(noise, (0, 0, 0), (255, 255, 255))

    cv2.imshow("Noise", noise)

    image = cv2.addWeighted(image, 0.75, noise, 0.25, 0)
    cv2.imshow("Noisy img", image)

    cv2.waitKey(0)
    #image = flip(image)
    #cv2.imshow("Flipped image", image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

