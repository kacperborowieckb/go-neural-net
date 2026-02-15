package main

import (
	"fmt"
)

func main() {
	matrixA := NewMatrix(2, 3, []float64{2, 3, 4, 5, 6, 7})
	matrixB := NewMatrix(3, 2, []float64{1, 2, 3, 4, 5, 6})
	//  2 3 4  1 2  31 40
	//  5 6 7  3 4  58 76
	//         5 6

	fmt.Println("SOME: ", Multiply(matrixA, matrixB))
}
