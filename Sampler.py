import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, patchsize):
        assert(len(train_images_list) == len(gt_segmentation_maps_list))
        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        #self.class_colors = classes_colors
        self.patchsize = patchsize

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        patches = []
        labels = []

        for index_train_image in np.arange(0, len(self.train_images_list)):
            train_img = mpimg.imread(self.train_images_list[index_train_image])
            label_img = mpimg.imread(
                self.gt_segmentation_maps_list[index_train_image])
            print("image size: " + str(train_img.shape))
            # we don't care about last pixels of photos
            patches_in_x = int(train_img.shape[1] / self.patchsize)
            patches_in_y = int(train_img.shape[0] / self.patchsize)
            for x in np.arange(patches_in_x):
                for y in np.arange(patches_in_y):
                    patch_img = train_img[y * self.patchsize:(y + 1) * self.patchsize,
                                          x*self.patchsize: (x+1)*self.patchsize]
                    patch_label = label_img[y * self.patchsize:(y + 1) * self.patchsize,
                                            x*self.patchsize: (x+1)*self.patchsize]
                    patches.append(patch_img)
                    labels.append(patch_label)

        return patches, labels
