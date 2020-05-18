# MNIST-Digit-Prediction

# Contents
This repository contains an implementation of an Artificial Neural Network ("NeuralNetwork3.py") designed to correctly recognize images 
of single hadnwritten digits. It is built to accept .csv representations of images. Included in this repo is a utility program 
("mnist_csv3.py") to convert an archive named "mnist.pkl.gz" into the appropriate csv format. Such a directory must contain training data, 
validation data, and test data. It will output separate csv's for training images, test images, and corresponding training and test 
labels.

In addition, this repo contains a sample set of relevant csv's ("test_image.csv", "test_label.csv", "train_image.csv", and "train_label.csv")
so that it may be demo'd without having to locate a correctly-formatted set of images and labels from MNIST. 

# Summary
This artificial neural network has two hidden layers built from perceptron neurons. To protect against over-fitting, each layer is 
built with a fraction of the number of neurons in the input layer. The hidden layer neurons make use of a Sigmoid activation function
to generate their respective outputs during the feed-forward stage. The output layer uses Softmax to select and scale the 
maximum-likelihood output of each neuron. It then calculates cross-entropy and propogates adjustments back through the layers,
updating weights and biases according to the learning rate.

Learning rate is initially quite high-- .5, and is then reduced exponentially with the numnber of examples trained. Accuracy 
converges after 15-30 epochs, depending on the size of the training set.

The output of this program is a csv file ("test_predictions.csv") of predictions for "test_image.csv".

# Input format
NeuralNetwork3.py takes as input three csv files of two types. 

The first type is a csv representing images. Each line corresponds to
a single image, where each value in the line is an integer from 0-255 signifying the color of that pixel. Each line contains 784
digits, since each image is expected to be 28x28. 

The second type of csv file is a labels file. This contains the labels for the corresponding iamge file. The correct format is a
series of single-digit rows, where each row signifies the correct value of the image represented by the corresponding row in the 
image csv. The image file and the label file should have the same number of rows. 

# Run instructions
This project requires Python3 as well as the Numpy package. After installing and setting up those, open a unix-style terminal and
navigate to the directory where you have downloaded this repository. The program takes optional commandline arguments of the paths
to the input files. If any of these options are not submitted, the program will assume the following pathnames:

    "./train_image.csv"
    "./train_label.csv"
    "./test_image.csv"

Run the following:

    python3 NeuralNetwork3.py [training images path] [training labels path] [test images path]
    
For a training set of 60,000 or fewer, the program should take under ten mintues.

