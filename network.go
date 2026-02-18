package main

import (
	"encoding/gob"
	"errors"
	"math"
	"math/rand"
	"os"
)

var (
	ErrFailedToSaveNetworkFile = errors.New("failed to save neural network to a file")
	ErrFailedToLoadNetworkFile = errors.New("failed to load neural network from a file")
)

type ActivationFn func(float64) float64

type NeuralNetwork struct {
	InputLayerSize  int
	HiddenLayerSize int
	OutputLayerSize int

	LearningRate float64

	HiddenLayer *Layer
	OutputLayer *Layer
}

func NewNeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize int, learningRate float64) *NeuralNetwork {
	return &NeuralNetwork{
		InputLayerSize:  inputLayerSize,
		HiddenLayerSize: hiddenLayerSize,
		OutputLayerSize: outputLayerSize,

		LearningRate: learningRate,

		HiddenLayer: &Layer{
			Weights: initializeWeights(hiddenLayerSize, inputLayerSize),
			Biases:  NewMatrix(hiddenLayerSize, 1, make([]float64, hiddenLayerSize)),
		},

		OutputLayer: &Layer{
			Weights: initializeWeights(outputLayerSize, hiddenLayerSize),
			Biases:  NewMatrix(outputLayerSize, 1, make([]float64, outputLayerSize)),
		},
	}
}
func (net *NeuralNetwork) Predict(inputData Vector) int {
	hiddenLayerOutput := net.HiddenLayer.ForwardPassLayer(inputData, Sigmoid)
	outputLayerOutput := net.OutputLayer.ForwardPassLayer(hiddenLayerOutput.Data, Sigmoid)

	predictionVector := outputLayerOutput.ColVector(0)

	currentProbability := 0.0
	predictionIndex := 0

	for i := range predictionVector {
		if predictionVector[i] > currentProbability {
			currentProbability = predictionVector[i]
			predictionIndex = i
		}
	}

	return predictionIndex
}

func (net *NeuralNetwork) Train(inputData, targetData Vector) {
	inputMatrix := NewMatrix(len(inputData), 1, inputData)
	targetMatrix := NewMatrix(len(targetData), 1, targetData)

	// pass data through network
	hiddenLayerOutput := net.HiddenLayer.ForwardPassLayer(inputData, Sigmoid)
	outputLayerOutput := net.OutputLayer.ForwardPassLayer(hiddenLayerOutput.Data, Sigmoid)

	// fmt.Println()
	// fmt.Println("OUTPUT:")
	// outputLayerOutput.Print()
	// fmt.Println()
	// fmt.Println("TARGET:")
	// fmt.Println(targetData)
	// fmt.Println()

	// output layer
	// error after activation
	outputLayerError := CalculateMeanSquaredErrorDerivative(outputLayerOutput, targetMatrix)
	// derivative of activations, how sensitive change is
	rawOutputLayerError := CalculateSigmoidDerivative(outputLayerOutput)
	// to calculate "blame" for the neuron - how much it needs to change
	chainedOutputError := outputLayerError.MultiplyElementWise(rawOutputLayerError)

	// how much to shift weights
	outputLayerGradient := chainedOutputError.Multiply(hiddenLayerOutput.Transpose())

	updatedOutputLayerWeights := net.OutputLayer.Weights.Subtract(outputLayerGradient.Scale(net.LearningRate))
	updatedOutputLayerBiases := net.OutputLayer.Biases.Subtract(chainedOutputError.Scale(net.LearningRate))

	// hidden layer
	// it's like outputLayerError for hidden layer
	distributedError := net.OutputLayer.Weights.Transpose().Multiply(chainedOutputError)

	rawHiddenLayerError := CalculateSigmoidDerivative(hiddenLayerOutput)
	chainedHiddenError := distributedError.MultiplyElementWise(rawHiddenLayerError)
	hiddenLayerGradient := chainedHiddenError.Multiply(inputMatrix.Transpose())

	updatedHiddenLayerWeights := net.HiddenLayer.Weights.Subtract(hiddenLayerGradient.Scale(net.LearningRate))
	updatedHiddenLayerBiases := net.HiddenLayer.Biases.Subtract(chainedHiddenError.Scale(net.LearningRate))

	// update network weights and biases
	net.OutputLayer.Weights = updatedOutputLayerWeights
	net.OutputLayer.Biases = updatedOutputLayerBiases

	net.HiddenLayer.Weights = updatedHiddenLayerWeights
	net.HiddenLayer.Biases = updatedHiddenLayerBiases
}

func (net *NeuralNetwork) Save(fileName string) {
	file, err := os.Create(fileName)
	if err != nil {
		panic(ErrFailedToSaveNetworkFile)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	err = encoder.Encode(net)
	if err != nil {
		panic(ErrFailedToSaveNetworkFile)
	}
}

func LoadNeuralNetwork(fileName string) *NeuralNetwork {
	file, err := os.Open(fileName)
	if err != nil {
		panic(ErrFailedToLoadNetworkFile)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	net := &NeuralNetwork{}

	err = decoder.Decode(net)
	if err != nil {
		panic(ErrFailedToLoadNetworkFile)
	}

	return net
}

type Layer struct {
	Weights *Matrix
	Biases  *Matrix
}

func (l *Layer) ForwardPassLayer(inputData Vector, activationFn ActivationFn) *Matrix {
	inputMatrix := NewMatrix(len(inputData), 1, inputData)

	return l.Weights.Multiply(inputMatrix).Add(l.Biases).Apply(activationFn)
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
