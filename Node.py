import numpy as np


class Node():
    def __init__(self, patches, labels, current_depth, tree_param):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}

        self.current_depth = current_depth + 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("CURENT DEPTH" + str(self.current_depth))

        self.patches = patches
        self.labels = labels

        self.max_depth = tree_param['depth']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.random_color_values = tree_param['random_color_values']
        self.pixel_locations = tree_param['pixel_locations']

        self.probabilities = np.zeros((len(self.classes), 1))
        self.tree_params = tree_param

        self.thresholds = np.linspace(0, 255, num=self.no_of_thresholds)
        self.patch_dim = self.patches[0].shape[0]
        self.colors = [0, 1, 2]

    # Function to create a new split nodez
    # provide your implementation

    def create_splitNode(self, leftchild, rightchild, feature):
        self.leftChild = leftchild
        self.rightChild = rightchild
    # Function to create a new leaf node
    # provide your implementation

    def create_leafNode(self, labels, classes):
        self.type = 'Leaf'
        self.leftChild = -1
        self.rightChild = -1
        pixel_location = self.feature["pixel_location"]
        label_pixel_location = np.zeros(
            (len(labels), 1), dtype=int)
        for label_indx, label in enumerate(labels):
            label_pixel_location[label_indx] = label[pixel_location[0],
                                                     pixel_location[1]]
        for i in self.classes:
            self.probabilities[i] = np.count_nonzero(
                label_pixel_location == i) / len(label_pixel_location)
        return

    # feel free to add any helper functions
    def run(self):
        if (self.max_depth + 1 < self.current_depth):
            self.create_leafNode(self.labels, self.classes)
            return
        left_labels, left_patches, right_labels, right_patches = self.best_split(
            self.patches, self.labels)
        if (len(left_labels) - self.minimum_patches_at_leaf < 0):
            self.create_leafNode(self.labels, self.classes)
            return
        if (len(right_labels) - self.minimum_patches_at_leaf < 0):
            self.create_leafNode(self.labels, self.classes)
            return
        else:
            self.leftChild = Node(left_patches, left_labels,
                                  self.current_depth, self.tree_params)
            self.leftChild.run()
            self.rightChild = Node(right_patches, right_labels,
                                   self.current_depth, self.tree_params)
            self.rightChild.run()

    def getFeatureResponse(self, patches, feature):
        random_pixel = self.generate_random_pixel_location()
        responses = np.zeros((len(patches)))
        for num_patch, patch in enumerate(patches):
            responses[num_patch] = patch[random_pixel[0],
                                         random_pixel[1], feature]
        return responses, random_pixel

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        left_split_index = np.argwhere(responses < threshold)
        right_split_index = np.argwhere(responses >= threshold)
        return left_split_index.transpose().tolist()[0], right_split_index.transpose().tolist()[0]

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch

    def generate_random_pixel_location(self):
        x = np.random.randint(self.patch_dim)
        y = np.random.randint(self.patch_dim)
        return np.array([x, y])

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        H = np.zeros(self.classes.shape[0])

        probabilities = np.zeros(self.classes.shape[0])
        for label_class in self.classes:
            count = np.sum(labels == label_class)
            print("for: " + str(label_class) + " I found " + str(count))
            probability = count / labels.shape[0]
            if (count == 0):
                H_class = 0
            else:
                H_class = probability * np.log(probability)
            H[label_class] = H_class
            probabilities[label_class] = probability
        return -np.sum(H)

    # Function to measure information gain for a given split
    # provide your implementation

    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - Entropyleft * (Nleft/Nall) - Entropyright * (Nright/Nall)

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        max_information_gain = 0
        threshold_max_information_gain = 0
        color_max_information_gain = 0
        pixel_location_max_information_gain = np.array([0, 0])

        bin_color_tests = np.random.choice(
            self.colors, self.random_color_values)
        for _ in np.arange(self.pixel_locations):
            for bin_color_test in bin_color_tests:
                responses, pixel_location = self.getFeatureResponse(
                    patches, bin_color_test)
                label_random_pixel_location = np.zeros(
                    (len(responses), 1), dtype=int)
                for label_indx, label in enumerate(labels):
                    label_random_pixel_location[label_indx] = label[pixel_location[0],
                                                                    pixel_location[1]]

                for threshold in self.thresholds:
                    entropy = self.compute_entropy(label_random_pixel_location)
                    left_split_index, right_split_index = self.getsplit(
                        responses, threshold)
                    print("Left size has: " + str(
                        len(left_split_index)) + " right side has: " + str(len(right_split_index)))
                    if (len(left_split_index) == 0 or len(right_split_index) == 0):
                        print("There is no point to split")
                        continue
                    left_entropy = self.compute_entropy(
                        label_random_pixel_location[left_split_index])
                    print("Left entropy: " + str(left_entropy))
                    right_entropy = self.compute_entropy(
                        label_random_pixel_location[right_split_index])
                    print("Right entropy: " + str(right_entropy))
                    information_gain = self.get_information_gain(
                        left_entropy, right_entropy,
                        entropy, len(label_random_pixel_location),
                        len(left_split_index), len(right_split_index))
                    print("Information gain: " + str(information_gain))
                    if (information_gain > max_information_gain):
                        print("Max_information gain exceeded, replacement")
                        print("was: " + str(max_information_gain))
                        print("now: " + str(information_gain))
                        max_information_gain = information_gain
                        max_information_gain_left_split_index = left_split_index
                        max_information_gain_right_split_index = right_split_index
                        threshold_max_information_gain = threshold
                        color_max_information_gain = bin_color_test
                        pixel_location_max_information_gain = pixel_location
        self.feature['color'] = color_max_information_gain
        self.feature['pixel_lication'] = pixel_location_max_information_gain
        self.feature['th'] = threshold_max_information_gain

        return_left_labels = []
        return_left_patches = []
        return_right_labels = []
        return_right_patches = []

        for left_index in max_information_gain_left_split_index:
            return_left_labels.append(labels[left_index])
            return_left_patches.append(patches[left_index])

        for right_index in max_information_gain_right_split_index:
            return_right_labels.append(labels[right_index])
            return_right_patches.append(patches[right_index])

        return return_left_labels, return_left_patches, return_right_labels, return_right_patches

    # feel free to add any helper functions
