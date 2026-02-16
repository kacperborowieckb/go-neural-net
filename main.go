package main

import "fmt"

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

func main() {
	matrixA := NewMatrix(2, 3, []float64{2, 3, 4, 5, 6, 7})
	matrixB := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	//  2 3 4  1 2  31 40
	//  5 6 7  3 4  58 76
	//         5 6

	_ = matrixA
	_ = matrixB

	net := NewNeuralNetwork(2, 3, 2)
	fmt.Println("hidden layer: ")
	net.hiddenLayer.weights.Print()
	fmt.Println("output layer: ")
	net.outputLayer.weights.Print()

	fmt.Println("")
	fmt.Println("Forward pass...")
	fmt.Println("")
	result := net.hiddenLayer.ForwardPassLayer([]float64{2.2, 4.4}, ReLU)
	result.Print()

}
