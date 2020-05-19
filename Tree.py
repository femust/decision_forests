import numpy as np
from Node import Node


class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.tree_params = tree_param
        self.nodes = []

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param

    def train(self):
        current_depth = 0
        self.node = Node(self.patches, self.labels,
                         current_depth, self.tree_params)
        self.node.run()

        # Function to predict probabilities for single image
        # provide your implementation
        # should return predicted class for every pixel in the test image

    def predict(self, I):
        pass

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
