# Neural Network From Scratch in Go.

Basic neural network implementation written in Go. Everything - including matrix operations - were implemented from scratch without using any external libraries.

No code in this project was generated using AI tools - it was built purely for learning purposes.

The network was trained and evaluated using the MNIST dataset of handwritten digits.

## Dataset

MNIST data sets are located at `data/mnist_sets.zip`. You must unzip the dataset before running the program.

## Pretrained Model

`models/mnist-model.gob` contains a pretrained neural network with **97% accuracy** on the MNIST test set.

## Usage

`go run . --train` - starts the training process and saves the trained model.

`go run . --evaluate` - evaluates the saved neural network using provided test dataset.
