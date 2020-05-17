import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patchsize):
        assert(len(train_images_list) == len(gt_segmentation_maps_list))
        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patchsize = patchsize

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        patches = []

        for index_train_image in np.arange(0, len(self.train_images_list)):
            train_img = mpimg.imread(self.train_images_list[index_train_image])
            label_img = mpimg.imread(
                self.gt_segmentation_maps_list[index_train_image])

            plt.imshow(train_img)
            plt.show()
            plt.imshow(label_img)
            plt.show()

            # feel free to add any helper functions
