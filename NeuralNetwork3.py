import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# dig = load_digits()
# onehot_target = pd.get_dummies(dig.target)
# x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)


def build_target(target: float) -> list:
    assert 0.0 <= target < 10.0, "out of bounds target. Must be == {0-9}"

    array = [0 for i in range(10)]
    array[int(target)] = 1
    return array


def load_train_images() -> list:
    images = np.genfromtxt("./train_image.csv", delimiter=",")
    return images


def load_train_labels() -> list:
    load = np.genfromtxt("./train_label.csv", delimiter=",")
    # vector = np.vectorize(build_target, otypes=[int])
    # labels = vector(load)
    labels = [build_target(i) for i in load]
    # print(labels)
    return labels

def load_test_images() -> list:
    images = np.genfromtxt("./test_image.csv", delimiter=",")
    return images


def load_test_labels() -> list:
    load = np.genfromtxt("./test_label.csv", delimiter=",")
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


class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 1568
        self.lr = 0.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.ones((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.ones((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.ones((1, op_dim))
        self.y = y

    def feedforward(self):

        # print("images: ", self.x.shape)
        # print("weights1: ",self.w1.shape)

        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        # self.a1 = ReLU(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        # self.a2 = ReLU(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        # print(self.a3)
        # print(self.y)
        a3_delta = cross_entropy(self.a3, self.y)  # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2)  # w2
        # a2_delta = z2_delta * d_ReLU(self.a2)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1)  # w1
        # a1_delta = z1_delta * d_ReLU(self.a1)  # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()

    def update_batch(self, batch, labels):
        self.x = batch
        self.y = labels

    def adjust_lr(self, lr):
        self.lr = lr

def output_plots(accuracies, learning_rates, epochs):
    # print(accuracies)
    plt.figure(1)
    plt.plot(epochs, accuracies)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.figure(2)
    plt.plot(epochs, learning_rates)
    plt.ylabel("LR")
    plt.xlabel("Epoch")
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

    batches = int(images.shape[0] / 100)
    batch_size = 100
    epochs = 10

    accuracies = []
    epoch_list = []
    learning_rates = []

    model = MyNN(images[0:100] / 255.0, np.array(labels[0:100]))

    lr_sched = LrScheduler(.5, .1)


    for x in range(epochs):


        lr = lr_sched.schedule(x)
        model.adjust_lr(lr)
        print(model.lr)

        for i in range(batches):

            start_index = i * batch_size
            end_index = start_index + batch_size
            model.update_batch(images[start_index:end_index] / 255.0, labels[start_index:end_index])
            model.feedforward()
            model.backprop()

        accuracies.append(get_acc(test_images / 255, np.array(test_labels), model))
        epoch_list.append(x)
        learning_rates.append(model.lr)

    output_plots(accuracies, learning_rates, epoch_list)
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


if __name__ == "__main__":
    all_labels = load_train_labels()
    all_images = load_train_images()

    test_labels = load_test_labels()
    test_images = load_test_images()

    # t_images = test_images[0:10000]
    # t_labels = test_labels[0:10000]
    #
    # images = all_images[0:10000]
    # labels = all_labels[0:10000]

    my_model = train(all_images, np.array(all_labels), test_images, np.array(test_labels))
    # print(labels[0])
    # print(images[0])
    print("Test accuracy : ", get_acc(test_images / 255, np.array(test_labels), my_model))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    # print("Training accuracy : ", get_acc(all_images / 255, np.array(all_labels), my_model))