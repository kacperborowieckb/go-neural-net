package main

import (
	"flag"
	"fmt"
)

// PLAN FOR FORWARD PASS
//
// 1. define a Network constructor with initialization with random weights
// [x] - number of input neurons, hidden layer, output,
// [x] - initialize weight matrices with random values -0.1 - 0.1 - for hidden and output
// [x] - bias matrices (no learning rate for now)
//
// 2. forward pass
// [x] - hidden layer calculations - multiply hidden matrix by input vector(for batch size 1)
// [x] - apply bias and activation for hidden layer
// [x] - repeat for output layer

// PLAN FOR TRAINING
//

func main() {
	train := flag.Bool("train", false, "train network")
	evaluate := flag.Bool("evaluate", false, "evaluate network")

	flag.Parse()

	if *train {
		fmt.Println("training")
		net := NewNeuralNetwork(784, 200, 10, 0.1)

		trainMnistDataset(net, 10)
		net.Save("models/mnist-model.gob")
	} else if *evaluate {
		net := LoadNeuralNetwork("models/mnist-model.gob")
		evaluateMnistDataset(net)
	} else {
		fmt.Println("please provide 'train' or 'evaluate' flag")
	}
}

func trainMnistDataset(net *NeuralNetwork, epochs int) {
	data := ReadMnistData("data/mnist_train.csv")

	for range epochs {
		for i := range data {
			target := make([]float64, 10)

			for j := range len(target) {
				target[j] = 0.01
			}

			target[data[i].label] = 0.99

			net.Train(data[i].input, target)
		}
	}
}

func evaluateMnistDataset(net *NeuralNetwork) {
	data := ReadMnistData("data/mnist_test.csv")
	correctPredictions := 0

	for i := range data {
		prediction := net.Predict(data[i].input)

		if prediction == data[i].label {
			correctPredictions++
		}

		fmt.Println("prediction: ", prediction, ", actual: ", data[i].label)
	}

	fmt.Println("accuracy: ", float64(correctPredictions)/float64(len(data)))
}
