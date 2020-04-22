import numpy as np
import matplotlib.pyplot as plt
import time
from math import ceil
import argparse
import sys
from scipy.special import softmax

start_time = time.time()

# dig = load_digits()
# onehot_target = pd.get_dummies(dig.target)
# x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)


def build_target(target: float) -> list:
    assert 0.0 <= target < 10.0, "out of bounds target. Must be == {0-9}"

    array = [0 for i in range(10)]
    array[int(target)] = 1
    return array


def load_train_images(path) -> list:
    images = np.genfromtxt(path, delimiter=",")
    return images


def load_train_labels(path) -> list:
    load = np.genfromtxt(path, delimiter=",")
    # vector = np.vectorize(build_target, otypes=[int])
    # labels = vector(load)
    labels = [build_target(i) for i in load]
    # print(labels)
    return labels

def load_test_images(path) -> list:
    images = np.genfromtxt(path, delimiter=",")
    return images


def load_test_labels(path = "./test_label.csv") -> list:
    load = np.genfromtxt(path, delimiter=",")
    # vector = np.vectorize(build_target, otypes=[int])
    # labels = vector(load)
    labels = [build_target(i) for i in load]
    # print(labels)
    return labels

# def ReLU(s):
#     s[s < 0] = 0
#     return s
#
# def binary_choice(i: float):
#     if i > 0:
#         return 1
#     else:
#         return 0
#
# def d_ReLU(s):
#     derived = np.array(s.shape)
#     v_func = np.vectorize(binary_choice)
#     derived = v_func(s)
#
#     return derived


def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))


def sigmoid_derivative(matrix):
    return matrix * (1 - matrix)


def softmax(matrix):

    max_prob = np.max(matrix, axis=1, keepdims=True)
    numerators = np.exp(matrix - max_prob)
    res = numerators / np.sum(numerators, axis=1, keepdims=True)

    return res


def cross_entropy(theoretical, actual):

    samples = actual.shape[0]
    res = theoretical - actual

    #normalize
    ans = res/samples

    return ans


class LrScheduler:
    class __LrScheduler:
        def __init__(self, init, init_k):
            self.init_lr = init
            self.k = init_k

        def schedule(self, e):
            lr = self.init_lr * np.exp(-self.k*e)
            return lr

    instance = None

    def __init__(self, init, init_k):
        if not LrScheduler.instance:
            LrScheduler.instance = LrScheduler.__LrScheduler(init, init_k)
        else:
            return

    @staticmethod
    def schedule(e):
        return LrScheduler.instance.schedule(e)


class NeuralNetwork:
    def __init__(self, input_I, input_L):

        self.lr = 0.5

        self.input_labels = input_L
        self.input_images = input_I

        input_nodes = input_I.shape[1]
        output_nodes = input_L.shape[1]
        hidden_neurons_1 = int(input_nodes/4)
        hidden_neurons_2 = int(input_nodes/8)

        # initialize weight matrices
        self.weights_layer1 = np.random.randn(input_nodes, hidden_neurons_1)
        self.weights_layer2 = np.random.randn(hidden_neurons_1, hidden_neurons_2)
        self.weights_layer3 = np.random.randn(hidden_neurons_2, output_nodes)

        #initialize biases
        self.bias_layer1 = np.ones((1, hidden_neurons_1))
        self.bias_layer2 = np.ones((1, hidden_neurons_2))
        self.bias_layer3 = np.ones((1, output_nodes))



    def feed_forward(self):

        # print("images: ", self.x.shape)
        # print("weights1: ",self.w1.shape)

        # x1 = np.dot(self.input_images, self.weights_layer1) + self.bias_layer1
        self.o_layer1 = sigmoid(np.dot(self.input_images, self.weights_layer1) + self.bias_layer1)

        # self.a1 = ReLU(z1)
        # x2 = np.dot(self.o_layer1, self.weights_layer2) + self.bias_layer2
        self.o_layer2 = sigmoid(np.dot(self.o_layer1, self.weights_layer2) + self.bias_layer2)

        # self.a2 = ReLU(z2)
        # x3 = np.dot(self.o_layer2, self.weights_layer3) + self.bias_layer3
        self.o_layer3 = softmax(np.dot(self.o_layer2, self.weights_layer3) + self.bias_layer3)

    def back_propagate(self):
        # loss = error(self.o_layer3, self.input_labels)
        # print('Error :', loss)
        # print(self.a3)
        # print(self.y)
        o_layer3_del = cross_entropy(self.o_layer3, self.input_labels)

        # x_layer2_delta = np.dot(o_layer3_delta, self.weights_layer3.T)
        o_layer2_del = np.dot(o_layer3_del, self.weights_layer3.T) * sigmoid_derivative(self.o_layer2)

        # x_layer1_delta = np.dot(o_layer2_delta, self.weights_layer2.T)
        o_layer1_del = np.dot(o_layer2_del, self.weights_layer2.T) * sigmoid_derivative(self.o_layer1)

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

    def predict(self, data):
        self.input_images = data
        self.feed_forward()
        return self.o_layer3.argmax(axis=1)
        #return self.o_layer3.argmax()

    def update_batch(self, batch, labels):
        self.input_images = batch
        self.input_labels = labels

    def adjust_lr(self, lr):
        self.lr = lr

def output_plots(accuracies, epochs):
    #learning_rates,
    # print(accuracies)
    plt.figure(1)
    plt.plot(epochs, accuracies)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    # plt.figure(2)
    # plt.plot(epochs, learning_rates)
    # plt.ylabel("LR")
    # plt.xlabel("Epoch")
    plt.show()


# def train(images, labels, test_images, test_labels):
def train(images, labels, test_images):

    batch_size = 10
    epochs = 30
    num_examples = images.shape[0]
    examples_trained = 0
    adjustments = 0

    batches = int(ceil(num_examples / batch_size))

    # accuracies = []
    # epoch_list = []
    # learning_rates = []

    model = NeuralNetwork(images[0:batch_size] / 255.0, labels[0:batch_size])

    lr_sched = LrScheduler(.5, .005)

    for x in range(epochs):

        for i in range(batches):

            #
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


        # accuracies.append(calculate_accuracy(np.array(test_images) / 255.0, np.array(test_labels), model))
        # epoch_list.append(x)
        # learning_rates.append(model.lr)

    # print("Test accuracy : ", calculate_accuracy(np.array(test_images) / 255.0, np.array(test_labels), model))
    results = model.predict(test_images / 255.0)
    output_prediction(results)
    # print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # output_plots(accuracies, epoch_list)

    return model


# def calculate_accuracy(x, y, model):
#     acc = 0
#     for xx, yy in zip(x, y):
#         s = model.predict(xx)
#         if s == np.argmax(yy):
#             acc += 1
#     return acc / len(x) * 100


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
    all_images = load_train_images(train_images_path)

    test_labels = load_test_labels()
    test_images = load_test_images(test_images_path)

    # t_images = test_images[0:10000]
    # t_labels = test_labels[0:10000]
    #
    # images = all_images[0:10000]
    # labels = all_labels[0:10000]

    train(np.array(all_images), np.array(all_labels), np.array(test_images))
    # train(np.array(all_images), np.array(all_labels), np.array(test_images), np.array(test_labels))
