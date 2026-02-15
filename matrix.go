package main

import (
	"errors"
)

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

type Vector []float64

func (m *Matrix) RowVector(row int) Vector {
	if m.Rows < row || row < 0 {
		panic(ErrInvalidDataLength)
	}

	start := m.Cols * row

	return m.Data[start : start+m.Cols]
}

func (m *Matrix) ColVector(col int) Vector {
	if m.Cols < col || col < 0 {
		panic(ErrInvalidDataLength)
	}

	data := make([]float64, m.Rows)

	for row := range m.Rows {
		data[row] = m.Data[row*m.Cols+col]
	}

	return data
}

func DotProduct(a, b Vector) float64 {
	sum := 0.0

	for i := range len(a) {
		sum += a[i] * b[i]
	}

	return sum
}

func Multiply(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(ErrInvalidMultiplyShape)
	}

	data := make([]float64, a.Rows*b.Cols)

	for row := range a.Rows {
		for col := range b.Cols {
			// wrong index here
			data[(row*b.Cols)+col] = DotProduct(a.RowVector(row), b.ColVector(col))
		}
	}

	return NewMatrix(a.Rows, b.Cols, data)
}
