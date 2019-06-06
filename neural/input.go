package neural

import "github.com/google/uuid"

type input struct {
	id          string
	value       float64
	outSynapses []*synapse
}

func newInput() *input {
	id, _ := uuid.NewRandom()
	return &input{id: id.String()}
}

func (i *input) propagate(value float64) {
	i.value = value
	for _, synapse := range i.outSynapses {
		synapse.propagate(i.value)
	}
}
