package neural

func connectLayerToLayer(from *layer, to *layer) {
	for _, fromNeuron := range from.neurons {
		for _, toNeuron := range to.neurons {
			connectNeuronToNeuron(fromNeuron, toNeuron)
		}
	}
}

func connectInputsToLayer(from []*input, to *layer) {
	for _, fromInput := range from {
		for _, toNeuron := range to.neurons {
			connectInputToNeuron(fromInput, toNeuron)
		}
	}
}

func connectNeuronToNeuron(from *neuron, to *neuron) {
	synapse := newSynapse()
	from.outSynapses = append(from.outSynapses, synapse)
	to.inSynapses = append(to.inSynapses, synapse)
}

func connectInputToNeuron(from *input, to *neuron) {
	synapse := newSynapse()
	from.outSynapses = append(from.outSynapses, synapse)
	to.inSynapses = append(to.inSynapses, synapse)
}
