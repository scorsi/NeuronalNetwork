package neural

type network struct {
	inputs       []*input
	outputsLayer *layer
	layers       []*layer
	Out          []float64
}

func NewNetwork(nbInputs, nbOutputs int, nbNeuronsPerLayers []int) *network {
	n := &network{}
	n.initInputs(nbInputs)
	n.initOutputs(nbOutputs)
	n.initLayers(nbNeuronsPerLayers)
	return n
}

func (n *network) initInputs(nbInputs int) {
	for ; nbInputs > 0; nbInputs-- {
		input := newInput()
		n.inputs = append(n.inputs, input)
	}
}

func (n *network) initOutputs(nbOutputs int) {
	n.outputsLayer = newLayer(nbOutputs)
}

func (n *network) initLayers(nbNeuronsPerLayers []int) {
	if len(nbNeuronsPerLayers) == 0 {
		connectInputsToLayer(n.inputs, n.outputsLayer)
	} else {
		for _, nb := range nbNeuronsPerLayers {
			layer := newLayer(nb)
			n.layers = append(n.layers, layer)
		}

		connectInputsToLayer(n.inputs, n.layers[0])
		if len(n.layers) > 1 {
			for i := 1; i < len(n.layers); i++ {
				connectLayerToLayer(n.layers[i-1], n.layers[i])
			}
		}
		connectLayerToLayer(n.layers[len(n.layers)-1], n.outputsLayer)
	}
}

func (n *network) setInputs(inputs []float64) {
	for i, input := range n.inputs {
		input.propagate(inputs[i])
	}
}

func (n *network) computeLayers() {
	for _, layer := range n.layers {
		layer.compute()
	}
	n.outputsLayer.compute()
}

func (n *network) generateOut() {
	n.Out = make([]float64, len(n.outputsLayer.neurons))

	for i, output := range n.outputsLayer.neurons {
		n.Out[i] = output.value
	}
}

func (n *network) Compute(inputs []float64) []float64 {
	n.setInputs(inputs)
	n.computeLayers()
	n.generateOut()
	return n.Out
}

func (n *network) learnLayers() {

}

func (n *network) Learn(outputs []float64) {

}
