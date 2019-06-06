package neural

import (
	"github.com/google/uuid"
	"math"
)

type neuron struct {
	id          string
	value       float64
	inSynapses  []*synapse
	outSynapses []*synapse
}

func newNeuron() *neuron {
	id, _ := uuid.NewRandom()
	return &neuron{id: id.String()}
}

func (n *neuron) compute() {
	var sum float64
	for _, s := range n.inSynapses {
		sum += s.outValue
	}
	n.value = 1.0 / (1.0 + math.Exp(-sum))
	for _, s := range n.outSynapses {
		s.propagate(n.value)
	}
}
