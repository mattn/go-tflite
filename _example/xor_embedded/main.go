package main

import (
	_ "embed"
	"fmt"
	"log"
	"math"

	"github.com/mattn/go-tflite"
)

//go:embed xor_model.tflite
var xor_model []byte

func main() {
	model := tflite.NewModel(xor_model)
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	for _, v := range [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}} {
		input := interpreter.GetInputTensor(0)
		input.SetFloat32s(v)
		interpreter.Invoke()
		output := interpreter.GetOutputTensor(0)
		got := output.Float32s()
		fmt.Printf("%d xor %d = %d\n", int(v[0]), int(v[1]), int(math.Ceil(float64(got[0]))))
	}
}
