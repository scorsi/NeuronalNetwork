package neural

import (
	"fmt"
	"math/rand"
)

type Dataset struct {
	Inputs  []float64
	Outputs []float64
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

func (nt *NetworkTrainer) backpropagation(dataset *Dataset) {
	deltas := make([][]float64, len(nt.network.layers)+1)

	last := len(nt.network.layers)
	l := nt.network.outputsLayer
	deltas[last] = make([]float64, len(l.neurons))
	for i, n := range l.neurons {
		deltas[last][i] = n.value * (1 - n.value) * (dataset.Outputs[i] - n.value)
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

func (nt *NetworkTrainer) shuffleDatasets(datasets []*Dataset) []*Dataset {
	nbdata := len(datasets)

	newDatasets := make([]*Dataset, nbdata)
	copy(newDatasets, datasets)

	currentIndex := nbdata

	for currentIndex > 0 {
		randomIndex := rand.Intn(nbdata - 1)
		currentIndex -= 1

		temporaryValue := newDatasets[currentIndex]
		newDatasets[currentIndex] = newDatasets[randomIndex]
		newDatasets[randomIndex] = temporaryValue
	}
	return newDatasets
}

func (nt *NetworkTrainer) Learn(datasets []*Dataset, debug bool) {
	nbdata := len(datasets)
	errors := nbdata
	for (float64(errors) / float64(nbdata)) > nt.DatasetErrorRate {
		errors = 0
		for _, d := range nt.shuffleDatasets(datasets) {
			results := nt.network.Compute(d.Inputs)
			if debug {
				fmt.Println("Inputs: ", d.Inputs)
				fmt.Println("Outputs: ", results)
				fmt.Println("Expected: ", d.Outputs)
				fmt.Println("-------------------------------------")
			}
			if nt.shouldLearn(results, d.Outputs) == true {
				errors += 1
				nt.backpropagation(d)
			}
		}
	}
}
