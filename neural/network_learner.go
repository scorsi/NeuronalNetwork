package neural

import (
	"fmt"
)

type Dataset struct {
	NbData  int
	Inputs  [][]float64
	Outputs [][]float64
}

type NetworkTrainer struct {
	network          *network
	DatasetErrorRate float64
	OutputErrorRate  float64
	Speed            float64
}

func NewNetworkTrainer(network *network) *NetworkTrainer {
	return &NetworkTrainer{network: network}
}

func (nt *NetworkTrainer) backpropagation(indexDataset int, dataset *Dataset) {
	deltas := make([][]float64, len(nt.network.layers)+1)

	last := len(nt.network.layers)
	l := nt.network.outputsLayer
	deltas[last] = make([]float64, len(l.neurons))
	for i, n := range l.neurons {
		deltas[last][i] = n.value * (1 - n.value) * (dataset.Outputs[indexDataset][i] - n.value)
	}

	for i := last - 1; i >= 0; i-- {
		l := nt.network.layers[i]
		deltas[i] = make([]float64, len(l.neurons))
		for j, n := range l.neurons {

			var sum float64 = 0
			for k, s := range n.outSynapses {
				sum += s.weight * deltas[i+1][k]
			}

			deltas[i][j] = n.value * (1 - n.value) * sum
		}
	}

	for i, l := range nt.network.layers {
		for j, n := range l.neurons {
			for _, s := range n.inSynapses {
				s.weight += nt.Speed * deltas[i][j] * s.inValue
			}
		}
	}
	for j, n := range nt.network.outputsLayer.neurons {
		for _, s := range n.inSynapses {
			s.weight += nt.Speed * deltas[last][j] * s.inValue
		}
	}
}

func (nt *NetworkTrainer) shouldLearn(outputs, expected []float64) bool {
	for i, got := range outputs {
		if got < expected[i]-nt.OutputErrorRate || got > expected[i]+nt.OutputErrorRate {
			return true
		}
	}
	return false
}

func (nt *NetworkTrainer) Learn(dataset *Dataset, debug bool) {
	errors := dataset.NbData
	for (float64(errors) / float64(dataset.NbData)) > nt.DatasetErrorRate {
		errors = 0
		for i := 0; i < dataset.NbData; i++ {
			outputs := nt.network.Compute(dataset.Inputs[i])
			if debug {
				fmt.Println("Inputs: ", dataset.Inputs[i])
				fmt.Println("Outputs: ", outputs)
				fmt.Println("Expected: ", dataset.Outputs[i])
				fmt.Println("-------------------------------------")
			}
			if nt.shouldLearn(outputs, dataset.Outputs[i]) == true {
				errors += 1
				nt.backpropagation(i, dataset)
			}
		}
	}
}
