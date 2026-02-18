package main

import (
	"encoding/csv"
	"errors"
	"os"
	"strconv"
)

var (
	ErrCannotOpenFile     = errors.New("cannot open file")
	ErrCannotReadFileData = errors.New("cannot read file data")
)

type MnistData struct {
	label int
	input []float64
}

func ReadMnistData(fileName string) []MnistData {
	file, err := os.Open(fileName)
	if err != nil {
		panic(ErrCannotOpenFile)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		panic(ErrCannotReadFileData)
	}

	convertedRecords := make([]MnistData, len(records)-1)

	// first row is column names
	for i := range len(records) - 1 {
		currentRecord := records[i+1]

		label, _ := strconv.Atoi(currentRecord[0])
		inputData := make([]float64, len(currentRecord)-1)

		for j := range len(currentRecord) - 1 {
			rawPixel, _ := strconv.ParseFloat(currentRecord[j+1], 64)

			inputData[j] = (rawPixel / 255.0 * 0.99) + 0.01
		}

		convertedRecords[i] = MnistData{
			label: label,
			input: inputData,
		}
	}

	return convertedRecords

}
