package main

import "errors"

var (
	ErrInvalidMatrixDimension = errors.New("matrix dimension is either negative or zero")
	ErrInvalidDataLength      = errors.New("data does not satisfy matrix dimensions")
	ErrInvalidMultiplyShape   = errors.New("incorrect matrix dimensions for dot product")
)

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func NewMatrix(rows, cols int, data []float64) *Matrix {
	if rows <= 0 || cols <= 0 {
		panic(ErrInvalidMatrixDimension)
	}

	if data != nil && len(data) != rows*cols {
		panic(ErrInvalidDataLength)
	}

	if data == nil {
		data = make([]float64, rows*cols)
	}

	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

func Multiply(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(ErrInvalidMultiplyShape)
	}

	data := make([]float64, a.Rows*b.Cols)

	return NewMatrix(a.Rows, b.Cols, data)
}
