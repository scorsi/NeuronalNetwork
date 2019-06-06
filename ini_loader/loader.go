package ini_loader

import (
	"fmt"
	"github.com/scorsi/number-net/neural"
	"gopkg.in/ini.v1"
	"os"
	"strconv"
)

func Load(path string) (int, int, *neural.Dataset) {
	f := openFile(path)
	nbInputs, nbOutputs, nbData := readHeader(f)
	inputs, outputs := readBody(f, nbInputs, nbOutputs, nbData)
	return nbInputs, nbOutputs, &neural.Dataset{NbData: nbData, Inputs: inputs, Outputs: outputs}
}

func openFile(path string) *ini.File {
	f, err := ini.Load(path)
	if err != nil {
		fmt.Printf("Fail to read file: %v", err)
		os.Exit(1)
	}
	return f
}

func readHeader(f *ini.File) (int, int, int) {
	var (
		nbInputs  int
		nbOutputs int
		nbData    int
	)
	nbInputs, err := f.Section("").Key("nbInputs").Int()
	if err != nil {
		fmt.Printf("Fail to read nbInputs in file: %v", err)
		os.Exit(1)
	}
	nbOutputs, err = f.Section("").Key("nbOutputs").Int()
	if err != nil {
		fmt.Printf("Fail to read nbOutputs in file: %v", err)
		os.Exit(1)
	}
	nbData, err = f.Section("").Key("nbData").Int()
	if err != nil {
		fmt.Printf("Fail to read nbData in file: %v", err)
		os.Exit(1)
	}
	return nbInputs, nbOutputs, nbData
}

func readBody(f *ini.File, nbInputs, nbOutputs, nbData int) ([][]float64, [][]float64) {
	inputs := make([][]float64, nbData)
	outputs := make([][]float64, nbData)
	for i := 0; i < nbData; i++ {
		inputs[i] = readBodySection(f, strconv.Itoa(i), "inputs", nbInputs)
		outputs[i] = readBodySection(f, strconv.Itoa(i), "outputs", nbOutputs)
	}
	return inputs, outputs
}

func readBodySection(f *ini.File, section, key string, max int) []float64 {
	if f.Section(section).HasKey(key) != true {
		fmt.Printf("Fail to read key %v in section %v", key, section)
		os.Exit(1)
	}
	data, err := f.Section(section).Key(key).StrictFloat64s(",")
	if err != nil {
		fmt.Printf("Fail to read %v in section %v in file: %v", key, section, err)
		os.Exit(1)
	}
	if len(data) != max {
		fmt.Printf("Invalid number of %v in section %v, expected %v but got %v", key, section, max, len(data))
		os.Exit(1)
	}
	return data
}
