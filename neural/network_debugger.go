package neural

import (
	"fmt"
)

type NetworkDebugger struct {
	network *network
}

func NewNetworkDebugger(network *network) *NetworkDebugger {
	return &NetworkDebugger{network: network}
}

func getSynapsesId(synapses []*synapse) []string {
	r := make([]string, len(synapses))
	for i, synapse := range synapses {
		r[i] = synapse.id
	}
	return r
}

func (nd *NetworkDebugger) debugInputsConnections() {
	for inputI, input := range nd.network.inputs {
		fmt.Println("Input(", inputI, ")[", input.id, "]")
		fmt.Println("	Nb Out =", len(input.outSynapses))
		fmt.Println("	Out =", getSynapsesId(input.outSynapses))
	}
}

func (nd *NetworkDebugger) debugLayersConnections() {
	for layerI, layer := range nd.network.layers {
		fmt.Println("Layer(", layerI, ")[", layer.id, "]")
		fmt.Println("	Nb Neurons =", len(layer.neurons))
		for neuronI, neuron := range layer.neurons {
			fmt.Println("	Neuron(", neuronI, ")[", neuron.id, "]")
			fmt.Println("		Nb In =", len(neuron.inSynapses))
			fmt.Println("		In =", getSynapsesId(neuron.inSynapses))
			fmt.Println("		Nb Out =", len(neuron.outSynapses))
			fmt.Println("		Out =", getSynapsesId(neuron.outSynapses))
		}
	}
}

func (nd *NetworkDebugger) debugOutputsConnections() {
	fmt.Println("OutputLayer[", nd.network.outputsLayer.id, "]")
	fmt.Println("	Nb Outputs =", len(nd.network.outputsLayer.neurons))
	for neuronI, neuron := range nd.network.outputsLayer.neurons {
		fmt.Println("	Output(", neuronI, ")[", neuron.id, "]")
		fmt.Println("		Nb In =", len(neuron.inSynapses))
		fmt.Println("		In =", getSynapsesId(neuron.inSynapses))
		fmt.Println("		Nb Out =", len(neuron.outSynapses))
		fmt.Println("		Out =", getSynapsesId(neuron.outSynapses))
	}
}

func (nd *NetworkDebugger) DebugConnections() {
	fmt.Println("---------[CONNECTIONS DEBUG]---------")
	fmt.Println("INPUTS:")
	nd.debugInputsConnections()
	fmt.Println("LAYERS:")
	nd.debugLayersConnections()
	fmt.Println("OUTPUTS:")
	nd.debugOutputsConnections()
	fmt.Println("-------------------------------------")
}

func (nd *NetworkDebugger) debugInputsSynapses() {
	for _, input := range nd.network.inputs {
		for _, synapse := range input.outSynapses {
			fmt.Println(synapse.id, "weight:", synapse.weight, "in:", synapse.inValue, "out:", synapse.outValue)
		}
	}
}

func (nd *NetworkDebugger) debugLayersSynapses() {
	for _, layer := range nd.network.layers {
		for _, neuron := range layer.neurons {
			for _, synapse := range neuron.outSynapses {
				fmt.Println(synapse.id, "weight:", synapse.weight, "in:", synapse.inValue, "out:", synapse.outValue)
			}
		}
	}
}

func (nd *NetworkDebugger) debugOutputsSynapses() {
	for _, neuron := range nd.network.outputsLayer.neurons {
		for _, synapse := range neuron.inSynapses {
			fmt.Println(synapse.id, "weight:", synapse.weight, "in:", synapse.inValue, "out:", synapse.outValue)
		}
	}
}

func (nd *NetworkDebugger) DebugSynapses() {
	fmt.Println("-----------[SYNAPSES DEBUG]----------")
	fmt.Println("INPUTS:")
	nd.debugInputsSynapses()
	fmt.Println("LAYERS:")
	nd.debugLayersSynapses()
	fmt.Println("OUTPUTS:")
	nd.debugOutputsSynapses()
	fmt.Println("-------------------------------------")
}

func (nd *NetworkDebugger) Debug() {
	nd.DebugConnections()
	nd.DebugSynapses()
}
