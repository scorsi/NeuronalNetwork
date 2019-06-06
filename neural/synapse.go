package neural

import (
	"github.com/google/uuid"
	"math/rand"
)

type synapse struct {
	id       string
	inValue  float64
	outValue float64
	weight   float64
}

func newSynapse() *synapse {
	id, _ := uuid.NewRandom()
	return &synapse{id: id.String(), weight: 2 * (rand.Float64() - 0.5)}
}

func (s *synapse) propagate(value float64) {
	s.inValue = value
	s.outValue = value * s.weight
}
