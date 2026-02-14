package main

import (
	"reflect"
	"testing"
)

func TestNewMatrix(t *testing.T) {
	tests := []struct {
		name        string
		rows        int
		cols        int
		data        []float64
		expected    *Matrix
		expectPanic error
	}{
		{
			name: "Valid matrix with provided data",
			rows: 2,
			cols: 2,
			data: []float64{1.1, 2.2, 3.3, 4.4},
			expected: &Matrix{
				Rows: 2,
				Cols: 2,
				Data: []float64{1.1, 2.2, 3.3, 4.4},
			},
			expectPanic: nil,
		},
		{
			name: "Valid matrix with nil data (allocates zeros)",
			rows: 2,
			cols: 3,
			data: nil,
			expected: &Matrix{
				Rows: 2,
				Cols: 3,
				Data: []float64{0, 0, 0, 0, 0, 0},
			},
			expectPanic: nil,
		},
		{
			name:        "Panic on negative rows",
			rows:        -1,
			cols:        2,
			data:        nil,
			expected:    nil,
			expectPanic: ErrInvalidMatrixDimension,
		},
		{
			name:        "Panic on zero columns",
			rows:        3,
			cols:        0,
			data:        nil,
			expected:    nil,
			expectPanic: ErrInvalidMatrixDimension,
		},
		{
			name:        "Panic on mismatched data length",
			rows:        2,
			cols:        2,
			data:        []float64{1, 2, 3},
			expected:    nil,
			expectPanic: ErrInvalidDataLength,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// scenarios for panic
			defer func() {
				r := recover()

				if tt.expectPanic != nil && r == nil {
					t.Errorf("expected panic %v, but did not panic", tt.expectPanic)
					return
				}

				if r != nil {
					err, ok := r.(error)
					if !ok || err != tt.expectPanic {
						t.Errorf("expected panic %v, got %v", tt.expectPanic, r)
					}
					return // success
				}
			}()

			got := NewMatrix(tt.rows, tt.cols, tt.data)

			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("NewMatrix(%d, %d) = %v, want %v", tt.rows, tt.cols, got, tt.expected)
			}
		})
	}

}
