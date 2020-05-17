from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def read_file(file_name):
    images = []
    labels = []
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if (i == 0):
                    number_images_classes = line.replace('\n', "").split(" ")
                    number_images_classes = list(
                        map(int, number_images_classes))
                if (str.startswith(line, "img")):
                    line = line.replace('\n', "")
                    line = line.split(" ")
                    images.append("./images/" + line[0])
                    labels.append("./images/" + line[1])
    except FileNotFoundError:
        print('File not found.')
    return images, labels, number_images_classes


def main():
    path_training_data = "./images/train_images.txt"
    #path_test_data = "./images/test_images.txt"
    images_train, labels_train, number_images_classes_train = read_file(
        path_training_data)

    print("Number of train imgs: " +
          str(number_images_classes_train[0]) + " number of classes : " + str(number_images_classes_train[1]))
    print("Train files")
    print(images_train)
    print("Labels train files")
    print(labels_train)
    class_colors = [0, 1, 2, 3]
    patch_size = 16
    patch_sampler = PatchSampler(
        images_train, labels_train, class_colors, patch_size)
    patches = patch_sampler.extractpatches()


# provide your implementation for the sheet 2 here

main()
