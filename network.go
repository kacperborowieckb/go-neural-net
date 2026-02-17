package main

import (
	"fmt"
	"math"
	"math/rand"
)

type ActivationFn func(float64) float64

type NeuralNetwork struct {
	inputLayerSize  int
	hiddenLayerSize int
	outputLayerSize int

	learningRate float64

	hiddenLayer *Layer
	outputLayer *Layer
}

func NewNeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize int, learningRate float64) *NeuralNetwork {
	return &NeuralNetwork{
		inputLayerSize:  inputLayerSize,
		hiddenLayerSize: hiddenLayerSize,
		outputLayerSize: outputLayerSize,

		learningRate: learningRate,

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

func (net *NeuralNetwork) Train(inputData, targetData Vector) {
	inputMatrix := NewMatrix(len(inputData), 1, inputData)
	targetMatrix := NewMatrix(len(targetData), 1, targetData)

	// pass data through network
	hiddenLayerOutput := net.hiddenLayer.ForwardPassLayer(inputData, Sigmoid)
	outputLayerOutput := net.outputLayer.ForwardPassLayer(hiddenLayerOutput.Data, Sigmoid)

	fmt.Println()
	outputLayerOutput.Print()
	fmt.Println()

	// output layer
	// error after activation
	outputLayerError := CalculateMeanSquaredErrorDerivative(outputLayerOutput, targetMatrix)
	// derivative of activations, how sensitive change is
	rawOutputLayerError := CalculateSigmoidDerivative(outputLayerOutput)
	// to calculate "blame" for the neuron - how much it needs to change
	chainedOutputError := outputLayerError.MultiplyElementWise(rawOutputLayerError)

	// how much to shift weights
	outputLayerGradient := chainedOutputError.Multiply(hiddenLayerOutput.Transpose())

	updatedOutputLayerWeights := net.outputLayer.weights.Subtract(outputLayerGradient.Scale(net.learningRate))
	updatedOutputLayerBiases := net.outputLayer.biases.Subtract(chainedOutputError.Scale(net.learningRate))

	// hidden layer
	// it's like outputLayerError for hidden layer
	distributedError := net.outputLayer.weights.Transpose().Multiply(chainedOutputError)

	rawHiddenLayerError := CalculateSigmoidDerivative(hiddenLayerOutput)
	chainedHiddenError := distributedError.MultiplyElementWise(rawHiddenLayerError)
	hiddenLayerGradient := chainedHiddenError.Multiply(inputMatrix.Transpose())

	updatedHiddenLayerWeights := net.hiddenLayer.weights.Subtract(hiddenLayerGradient.Scale(net.learningRate))
	updatedHiddenLayerBiases := net.hiddenLayer.biases.Subtract(chainedHiddenError.Scale(net.learningRate))

	// update network weights and biases
	net.outputLayer.weights = updatedOutputLayerWeights
	net.outputLayer.biases = updatedOutputLayerBiases

	net.hiddenLayer.weights = updatedHiddenLayerWeights
	net.hiddenLayer.biases = updatedHiddenLayerBiases
}

type Layer struct {
	weights *Matrix
	biases  *Matrix
}

func (l *Layer) ForwardPassLayer(inputData Vector, activationFn ActivationFn) *Matrix {
	inputMatrix := NewMatrix(len(inputData), 1, inputData)

	return l.weights.Multiply(inputMatrix).Add(l.biases).Apply(activationFn)
}

func Sigmoid(val float64) float64 {
	return 1.0 / (1 + math.Exp(-1*val))
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
