package main

func CalculateMeanSquaredError(predicted *Matrix, target *Matrix) float64 {
	meanSquaredErrorMatrix := predicted.Subtract(target).Apply(square)

	return meanSquaredErrorMatrix.Average()
}

func CalculateMeanSquaredErrorDerivative(predicted *Matrix, target *Matrix) *Matrix {
	return predicted.Subtract(target)
}

func CalculateSigmoidDerivative(matrix *Matrix) *Matrix {
	return matrix.Apply(sigmoidDerivative)
}

func square(val float64) float64 {
	return val * val
}

func sigmoidDerivative(val float64) float64 {
	return val * (1.0 - val)
}
