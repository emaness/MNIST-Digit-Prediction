import numpy as np
import matplotlib.pyplot as plt
import time
from math import ceil
import argparse

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

def ReLU(s):
    s[s < 0] = 0
    return s

def binary_choice(i: float):
    if i > 0:
        return 1
    else:
        return 0

def d_ReLU(s):
    derived = np.array(s.shape)
    v_func = np.vectorize(binary_choice)
    derived = v_func(s)

    return derived


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def softmax(s):
    # print(s)
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    # print(exps)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]

    # print(pred)
    # print(real)

    res = pred - real

    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    # print(real)
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


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
        self.input_images = input_I
        self.lr = 0.5
        input_nodes = input_I.shape[1]
        output_nodes = input_L.shape[1]
        hidden_neurons_1 = input_nodes*2
        hidden_neurons_2 = input_nodes*2

        self.weights_layer1 = np.random.randn(input_nodes, hidden_neurons_1)
        self.bias_layer1 = np.ones((1, hidden_neurons_1))
        self.weights_layer2 = np.random.randn(hidden_neurons_1, hidden_neurons_2)
        self.bias_layer2 = np.ones((1, hidden_neurons_2))
        self.weights_layer3 = np.random.randn(hidden_neurons_2, output_nodes)
        self.bias_layer3 = np.ones((1, output_nodes))
        self.input_labels = input_L

    def feed_forward(self):

        # print("images: ", self.x.shape)
        # print("weights1: ",self.w1.shape)

        x1 = np.dot(self.input_images, self.weights_layer1) + self.bias_layer1
        self.o_layer1 = sigmoid(x1)
        # self.a1 = ReLU(z1)
        x2 = np.dot(self.o_layer1, self.weights_layer2) + self.bias_layer2
        self.o_layer2 = sigmoid(x2)
        # self.a2 = ReLU(z2)
        x3 = np.dot(self.o_layer2, self.weights_layer3) + self.bias_layer3
        self.o_layer3 = softmax(x3)

    def back_propogate(self):
        # loss = error(self.o_layer3, self.input_labels)
        # print('Error :', loss)
        # print(self.a3)
        # print(self.y)
        o_layer3_delta = cross_entropy(self.o_layer3, self.input_labels)  # w3
        x_layer2_delta = np.dot(o_layer3_delta, self.weights_layer3.T)
        o_layer2_delta = x_layer2_delta * sigmoid_derv(self.o_layer2)  # w2
        # a2_delta = z2_delta * d_ReLU(self.a2)  # w2
        x_layer1_delta = np.dot(o_layer2_delta, self.weights_layer2.T)
        o_layer1_delta = x_layer1_delta * sigmoid_derv(self.o_layer1)  # w1
        # a1_delta = z1_delta * d_ReLU(self.a1)  # w1

        self.weights_layer3 -= self.lr * np.dot(self.o_layer2.T, o_layer3_delta)
        self.bias_layer3 -= self.lr * np.sum(o_layer3_delta, axis=0, keepdims=True)
        self.weights_layer2 -= self.lr * np.dot(self.o_layer1.T, o_layer2_delta)
        self.bias_layer2 -= self.lr * np.sum(o_layer2_delta, axis=0)
        self.weights_layer1 -= self.lr * np.dot(self.input_images.T, o_layer1_delta)
        self.bias_layer1 -= self.lr * np.sum(o_layer1_delta, axis=0)

    def predict(self, data):
        self.input_images = data
        self.feed_forward()
        return self.o_layer3.argmax(axis=1)

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


def train(images, labels, test_images, test_labels):

    # print(images.shape[0] / 100)
    # print(images.shape[1])

    # image_sets = np.empty([int(images.shape[0] / 100), images.shape[1]])
    # label_sets = np.empty([int(labels.shape[0] / 100), labels.shape[1]])
    # print(image_sets.shape)
    # print(label_sets.shape)

    # Batch size of 10,000
    # images1 = images[0:10000]
    # images2 = images[10000:20000]
    # images3 = images[20000:30000]
    # images4 = images[30000:40000]
    # images5 = images[40000:50000]
    # images6 = images[50000:60000]

    # Batch size of 5,000
    # images1 = images[0:5000]
    # images2 = images[5000:10000]
    # images3 = images[10000:15000]
    # images4 = images[15000:20000]
    # images5 = images[20000:25000]
    # images6 = images[25000:30000]
    # images7 = images[30000:35000]
    # images8 = images[35000:40000]
    # images9 = images[40000:45000]
    # images10 = images[45000:50000]
    # images11 = images[50000:55000]
    # images12 = images[55000:60000]


    # Batch size of 10,000
    # labels1 = labels[0:10000]
    # labels2 = labels[10000:20000]
    # labels3 = labels[20000:30000]
    # labels4 = labels[30000:40000]
    # labels5 = labels[40000:50000]
    # labels6 = labels[50000:60000]

    # Batch size of 5,000
    # labels1 = labels[0:5000]
    # labels2 = labels[5000:10000]
    # labels3 = labels[10000:15000]
    # labels4 = labels[15000:20000]
    # labels5 = labels[20000:25000]
    # labels6 = labels[25000:30000]
    # labels7 = labels[30000:35000]
    # labels8 = labels[35000:40000]
    # labels9 = labels[40000:45000]
    # labels10 = labels[45000:50000]
    # labels11 = labels[50000:55000]
    # labels12 = labels[55000:60000]
    #
    # for i in range(image_sets.shape[0]):
    #
    #     start_index = i*100
    #     end_index = start_index + 100
    #     image_sets[i] = images[start_index:end_index]
    #     label_sets[i] = labels[start_index:end_index]

    # get rid of magic numbers
    batch_size = 100
    epochs = 10
    num_examples = images.shape[0]

    batches = int(ceil(num_examples / batch_size))
    # print("batches: %d", batches)

    accuracies = []
    epoch_list = []
    learning_rates = []

    model = NeuralNetwork(images[0:batch_size] / 255.0, np.array(labels[0:batch_size]))

    lr_sched = LrScheduler(.5, .1)

    for x in range(epochs):

        lr = lr_sched.schedule(x)
        model.adjust_lr(lr)
        # print(model.lr)

        for i in range(batches):

            # print("batch number: %d" i)
            start_index = i * batch_size
            end_index = min(start_index + batch_size, num_examples)

            # print("current batch size: %d" % (end_index - start_index))

            # print("Start: %d  End: %d" % (start_index, end_index))

            model.update_batch(images[start_index:end_index] / 255.0, labels[start_index:end_index])
            model.feed_forward()
            model.back_propogate()

        accuracies.append(get_acc(test_images / 255, np.array(test_labels), model))
        epoch_list.append(x)
        # learning_rates.append(model.lr)

    # print("Test accuracy : ", get_acc(test_images / 255, np.array(test_labels), model))
    results = model.predict(test_images / 255)
    output_prediction(results)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    output_plots(accuracies, epoch_list)
    #learning_rates,
    # model.update_batch(images1 / 255.0, labels1)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images2 / 255.0, labels2)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images3 / 255.0, labels3)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images4 / 255.0, labels4)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images5 / 255.0, labels5)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images6 / 255.0, labels6)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images7 / 255.0, labels7)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images8 / 255.0, labels8)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images9 / 255.0, labels9)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images10 / 255.0, labels10)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images11 / 255.0, labels11)
    # model.feedforward()
    # model.backprop()
    #
    # model.update_batch(images12 / 255.0, labels12)
    # model.feedforward()
    # model.backprop()


    return model


def get_acc(x, y, model):
    acc = 0
    for xx, yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc += 1
    return acc / len(x) * 100


def output_prediction(results):
    with open("test_predictions.csv", 'w') as f:
        for x in results:
            print(x)
            line = str(x) + '\n'
            f.write(line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", type=str, help="Directory containing logs", )
    parser.add_argument("--train_labels", type=str, help="Directory containing logs", )
    parser.add_argument("--test_images", type=str, help="File of user subscriptions", )
    args = parser.parse_args()

    train_images_path = "./train_image.csv"
    train_labels_path = "./train_label.csv"
    test_images_path = "./test_image.csv"

    if args.train_images is not None:
        train_images_path = args.train_images

    if args.train_labels is not None:
        train_labels_path = args.train_images

    if args.test_images is not None:
        test_images_path = args.test_labels

    all_labels = load_train_labels(train_labels_path)
    all_images = load_train_images(train_images_path)

    test_labels = load_test_labels()
    test_images = load_test_images(test_images_path)

    # t_images = test_images[0:10000]
    # t_labels = test_labels[0:10000]
    #
    images = all_images[0:101]
    labels = all_labels[0:101]

    my_model = train(images, np.array(labels), test_images, np.array(test_labels))
    # print(labels[0])
    # print(images[0])
    # print("Training accuracy : ", get_acc(all_images / 255, np.array(all_labels), my_model))