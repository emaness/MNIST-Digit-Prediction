"""
Author: Everett Maness
Date: 5/17/2020
Written for CSCI 561 at University of Southern California
"""


import numpy as np
from math import ceil
import sys


def build_target(target: float) -> list:
    assert 0.0 <= target < 10.0, "out of bounds target. Must be == {0-9}"

    array = [0 for i in range(10)]
    array[int(target)] = 1
    return array


def load_images(path) -> list:
    images = np.genfromtxt(path, delimiter=",")
    return images


def load_train_labels(path) -> list:
    load = np.genfromtxt(path, delimiter=",")

    labels = [build_target(i) for i in load]

    return labels


def sigmoid(matrix) -> float:
    return 1 / (1 + np.exp(-matrix))


def sigmoid_derivative(matrix):
    return matrix * (1 - matrix)


def softmax(matrix) -> float:

    max_prob = np.max(matrix, axis=1, keepdims=True)
    numerators = np.exp(matrix - max_prob)
    res = numerators / np.sum(numerators, axis=1, keepdims=True)

    return res


"""
Finds the error to use in calculating the corrections for back-propogation
"""
def cross_entropy(theoretical, actual) -> float:

    samples = actual.shape[0]
    res = theoretical - actual

    # normalize
    ans = res/samples

    return ans

"""
Abstracts a learning rate manager. Implemented as a singleton to allow for possible multi-threading
in the future. 
"""
class LrScheduler:
    class __LrScheduler:
        def __init__(self, init, init_k):
            self.init_lr = init
            self.k = init_k

        def schedule(self, e) -> float:
            lr = self.init_lr * np.exp(-self.k*e)
            return lr

    instance = None

    def __init__(self, init, init_k):
        if not LrScheduler.instance:
            LrScheduler.instance = LrScheduler.__LrScheduler(init, init_k)
        else:
            return

    @staticmethod
    def schedule(e) -> float:
        return LrScheduler.instance.schedule(e)

"""
Description: Abstracts a Neural Network. 2 hidden layers, 1/4 and 1/8 the number of neurons
compared with input nodes, respectively.
"""
class NeuralNetwork:

    """
    input_I: an mxn array of input image pixel values corresponding to a set of images.
    input_L: an mx1 array of input image labels
    """
    def __init__(self, input_I, input_L):

        layer1_scaler = 4
        layer2_scaler = 8

        # set initial learning rate (to be updated later)
        self.lr = 0.5

        # set training image and label instance variables
        self.input_labels = input_L
        self.input_images = input_I

        # initialize dimensions of network
        input_nodes = input_I.shape[1]
        output_nodes = input_L.shape[1]
        hidden_neurons_1 = int(input_nodes/layer1_scaler)
        hidden_neurons_2 = int(input_nodes/layer2_scaler)

        # initialize weight matrices
        self.weights_layer1 = np.random.randn(input_nodes, hidden_neurons_1)
        self.weights_layer2 = np.random.randn(hidden_neurons_1, hidden_neurons_2)
        self.weights_layer3 = np.random.randn(hidden_neurons_2, output_nodes)

        # initialize biases
        self.bias_layer1 = np.ones((1, hidden_neurons_1))
        self.bias_layer2 = np.ones((1, hidden_neurons_2))
        self.bias_layer3 = np.ones((1, output_nodes))


    """
    feed forward inputs through the layers of perceptrons 
    """
    def feed_forward(self):
        self.o_layer1 = sigmoid(np.dot(self.input_images, self.weights_layer1) + self.bias_layer1)
        self.o_layer2 = sigmoid(np.dot(self.o_layer1, self.weights_layer2) + self.bias_layer2)
        self.o_layer3 = softmax(np.dot(self.o_layer2, self.weights_layer3) + self.bias_layer3)

    """
    execute back-propogation, update weights and biases accordingly
    """
    def back_propagate(self):

        # calculate update values
        o_layer3_del = cross_entropy(self.o_layer3, self.input_labels)
        o_layer2_del = np.dot(o_layer3_del, self.weights_layer3.T) * sigmoid_derivative(self.o_layer2)
        o_layer1_del = np.dot(o_layer2_del, self.weights_layer2.T) * sigmoid_derivative(self.o_layer1)

        # update weights and biases
        self.update_weights(o_layer1_del, o_layer2_del, o_layer3_del)
        self.update_biases(o_layer1_del, o_layer2_del, o_layer3_del)


    def update_weights(self, o_layer1_del, o_layer2_del, o_layer3_del):
        self.weights_layer3 = self.weights_layer3 - (self.lr * np.dot(self.o_layer2.T, o_layer3_del))
        self.weights_layer2 = self.weights_layer2 - (self.lr * np.dot(self.o_layer1.T, o_layer2_del))
        self.weights_layer1 = self.weights_layer1 - (self.lr * np.dot(self.input_images.T, o_layer1_del))


    def update_biases(self, o_layer1_del, o_layer2_del, o_layer3_del):
        self.bias_layer3 = self.bias_layer3 - (self.lr * np.sum(o_layer3_del, axis=0, keepdims=True))
        self.bias_layer2 = self.bias_layer2 - (self.lr * np.sum(o_layer2_del, axis=0))
        self.bias_layer1 = self.bias_layer1 - (self.lr * np.sum(o_layer1_del, axis=0))

    def predict(self, data) -> list:
        self.input_images = data
        self.feed_forward()
        return self.o_layer3.argmax(axis=1)

    def update_batch(self, batch, labels):
        self.input_images = batch
        self.input_labels = labels

    def adjust_lr(self, lr):
        self.lr = lr


def train(images, labels, test_images):

    batch_size = 10
    epochs = 30

    # number of images in training set
    num_examples = images.shape[0]

    examples_trained = 0
    adjustments = 0

    batches = int(ceil(num_examples / batch_size))

    model = NeuralNetwork(images[0:batch_size] / 255.0, labels[0:batch_size])

    #instantiate learning rate scheduler
    lr_sched = LrScheduler(.5, .005)

    for x in range(epochs):

        for i in range(batches):

            # choose end_index with variable-sized dataset in mind
            start_index = i * batch_size
            end_index = min(start_index + batch_size, num_examples)

            model.update_batch(images[start_index:end_index] / 255.0, labels[start_index:end_index])
            model.feed_forward()
            model.back_propagate()

            examples_trained += (end_index - start_index)

            # Update the learning rate every 60,000 examples trained
            if examples_trained >= 60000:
                lr = lr_sched.schedule(adjustments)
                model.adjust_lr(lr)
                adjustments += 1
                examples_trained = 0

    results = model.predict(test_images / 255.0)
    output_prediction(results)


def output_prediction(results):
    with open("test_predictions.csv", 'w') as f:
        for x in results:
            line = str(x) + '\n'
            f.write(line)


if __name__ == "__main__":

    train_images_path = "./train_image.csv"
    train_labels_path = "./train_label.csv"
    test_images_path = "./test_image.csv"

    if sys.argv[1] is not None:
        train_images_path = sys.argv[1]

    if sys.argv[2] is not None:
        train_labels_path = sys.argv[2]

    if sys.argv[3] is not None:
        test_images_path = sys.argv[3]

    all_labels = load_train_labels(train_labels_path)
    all_images = load_images(train_images_path)

    test_images = load_images(test_images_path)

    train(np.array(all_images), np.array(all_labels), np.array(test_images))
