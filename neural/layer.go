package neural

import "github.com/google/uuid"

type layer struct {
	id      string
	neurons []*neuron
}

func newLayer(neurons int) *layer {
	id, _ := uuid.NewRandom()
	l := &layer{id: id.String()}
	l.initNeurons(neurons)
	return l
}

func (l *layer) initNeurons(neurons int) {
	for ; neurons > 0; neurons-- {
		neuron := newNeuron()
		l.neurons = append(l.neurons, neuron)
	}
}

func (l *layer) compute() {
	for _, neuron := range l.neurons {
		neuron.compute()
	}
}
