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
        agumented_measurements.append(-1 * measurement)

    return agumented_dataset, agumented_measurements


if __name__ == '__main__':
    image_name = "data/IMG/left_2016_12_01_13_34_23_036.jpg"
    image = cv2.imread(image_name)
    cv2.imshow("Imported img", image)
    cv2.waitKey(0)
    image = flip(image)
    cv2.imshow("Flipped image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

