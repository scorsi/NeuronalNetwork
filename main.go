package main

import (
	"github.com/scorsi/number-net/ini_loader"
	"github.com/scorsi/number-net/neural"
)

func main() {
	nbInputs, nbOutputs, dataset := ini_loader.Load("./data/logical_all.ini")
	network := neural.NewNetwork(nbInputs, nbOutputs, []int{10})
	learner := neural.NewNetworkTrainer(network)
	debugger := neural.NewNetworkDebugger(network)
	learner.Speed = 0.1
	learner.DatasetErrorRate = 0.1
	learner.OutputErrorRate = 0.01
	learner.Learn(dataset, true)
	debugger.Debug()
}
