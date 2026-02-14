package main

import (
	"fmt"
)

func main() {
	matrix := NewMatrix(3, 2, []float64{2, 3, 4, 5, 6, 7})
	fmt.Println("SOME", matrix.ColVector(0))
}
