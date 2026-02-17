package main

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidMatrixDimension          = errors.New("matrix dimension is either negative or zero")
	ErrInvalidDataLength               = errors.New("data does not satisfy matrix dimensions")
	ErrInvalidMultiplyShape            = errors.New("incorrect matrix dimensions for dot product")
	ErrInvalidMultiplyElementWiseShape = errors.New("incorrect matrix dimensions for dot product")
	ErrInvalidOperationShape           = errors.New("incorrect matrix dimensions for current operation")
	ErrInvalidProductVectorsLength     = errors.New("vectors must be same length")
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
	if row >= m.Rows || row < 0 {
		panic(ErrInvalidDataLength)
	}

	start := m.Cols * row

	return m.Data[start : start+m.Cols]
}

func (m *Matrix) ColVector(col int) Vector {
	if m.Cols <= col || col < 0 {
		panic(ErrInvalidDataLength)
	}

	data := make([]float64, m.Rows)

	for row := range m.Rows {
		data[row] = m.Data[row*m.Cols+col]
	}

	return data
}

func (m *Matrix) Add(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(ErrInvalidOperationShape)
	}

	dataSize := m.Rows * m.Cols

	data := make([]float64, dataSize)

	for i := range dataSize {
		data[i] = m.Data[i] + other.Data[i]
	}

	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Subtract(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(ErrInvalidOperationShape)
	}

	dataSize := m.Rows * m.Cols

	data := make([]float64, dataSize)

	for i := range dataSize {
		data[i] = m.Data[i] - other.Data[i]
	}

	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Multiply(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic(ErrInvalidMultiplyShape)
	}

	data := make([]float64, m.Rows*other.Cols)

	for row := range m.Rows {
		for col := range other.Cols {
			data[(row*other.Cols)+col] = DotProduct(m.RowVector(row), other.ColVector(col))
		}
	}

	return NewMatrix(m.Rows, other.Cols, data)
}

func (m *Matrix) MultiplyElementWise(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(ErrInvalidMultiplyElementWiseShape)
	}

	dataSize := m.Rows * m.Cols
	data := make([]float64, dataSize)

	for i := range dataSize {
		data[i] = m.Data[i] * other.Data[i]
	}

	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	dataSize := m.Rows * m.Cols

	data := make([]float64, dataSize)

	for i := range dataSize {
		data[i] = fn(m.Data[i])
	}

	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Scale(rate float64) *Matrix {
	return m.Apply(func(val float64) float64 { return val * rate })
}

func (m *Matrix) Transpose() *Matrix {
	return NewMatrix(m.Cols, m.Rows, m.Data)
}

func (m *Matrix) Average() float64 {
	dataSize := m.Rows * m.Cols
	sum := 0.0

	for i := range dataSize {
		sum += m.Data[i]
	}

	return sum / float64(dataSize)
}

func (m *Matrix) Print() {
	for row := range m.Rows {
		fmt.Printf("| ")
		for col := range m.Cols {
			fmt.Printf("%8.4f ", m.Data[row*m.Cols+col])
		}
		fmt.Printf("|\n")
	}
}

func DotProduct(a, b Vector) float64 {
	if len(a) != len(b) {
		panic(ErrInvalidProductVectorsLength)
	}

	sum := 0.0

	for i := range len(a) {
		sum += a[i] * b[i]
	}

	return sum
}
