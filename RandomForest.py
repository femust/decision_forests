from Tree import DecisionTree
import numpy as np


class Forest():
    def __init__(self, patches, labels, tree_param, n_trees):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        for i in range(n_trees):
            self.trees.append(DecisionTree(self.patches, self.labels, self.tree_param))

        self.patchsize = 16
        self.classes = tree_param['classes']

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        for tree in self.trees:
            tree.train()
        return self.trees

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return class for every pixel in the test image
    def test(self, I):
        pixels_in_x = I.shape[1]
        pixels_in_y = I.shape[0]
        predict_image = np.zeros((I.shape[0], I.shape[1]))
        for x in np.arange(pixels_in_x-self.patchsize):
            for y in np.arange(pixels_in_y-self.patchsize):
                patch_img = I[y: y + self.patchsize,
                              x: x + self.patchsize]
                if (patch_img.shape[0] != 16 or patch_img.shape[1] != 16):
                    print("img patch" + str(patch_img.shape))
                mid_point_y = int(y + self.patchsize/2)
                mid_point_x = int(x + self.patchsize/2)
                ### Ensemble all probs here
                logits = np.zeros((4,1))
                for tree in self.trees:
                    logits += tree.predict_from_tree(patch_img)
                logits /= self.ntrees
                index = np.argmax(logits)
                predict_image[mid_point_y,
                              mid_point_x] = self.classes[index]
        return predict_image
        

    # feel free to add any helper functions
