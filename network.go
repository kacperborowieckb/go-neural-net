package main

import "math/rand"

type ActivationFn func(float64) float64

type NeuralNetwork struct {
	inputLayerSize  int
	hiddenLayerSize int
	outputLayerSize int

	hiddenLayer *Layer
	outputLayer *Layer
}

func NewNeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize int) *NeuralNetwork {
	return &NeuralNetwork{
		inputLayerSize:  inputLayerSize,
		hiddenLayerSize: hiddenLayerSize,
		outputLayerSize: outputLayerSize,

		hiddenLayer: &Layer{
			weights: initializeWeights(hiddenLayerSize, inputLayerSize),
			biases:  NewMatrix(hiddenLayerSize, 1, make([]float64, hiddenLayerSize)),
		},

		outputLayer: &Layer{
			weights: initializeWeights(outputLayerSize, hiddenLayerSize),
			biases:  NewMatrix(outputLayerSize, 1, make([]float64, outputLayerSize)),
		},
	}
}

type Layer struct {
	weights *Matrix
	biases  *Matrix
}

func (l *Layer) ForwardPassLayer(inputData Vector, activationFn ActivationFn) *Matrix {
	inputMatrix := NewMatrix(len(inputData), 1, inputData)

	return l.weights.Multiply(inputMatrix).Add(l.biases).Apply(activationFn)
}

func ReLU(val float64) float64 {
	if val < 0 {
		return 0
	}

	return val
}

func initializeWeights(rows, cols int) *Matrix {
	dataSize := rows * cols

	data := make([]float64, dataSize)

	for i := range dataSize {
		// random weights between -0.1 and 0.1
		data[i] = (rand.Float64() * 0.2) - 0.1
	}

	return NewMatrix(rows, cols, data)
}
