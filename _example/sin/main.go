package main

import (
	"fmt"
	"log"
	"math"

	"github.com/mattn/go-tflite"
)

func main() {
	model := tflite.NewModelFromFile("sin_model.tflite")
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	for i := -180; i < 180; i++ {
		v := float64(i) * math.Pi / 180.0
		input := interpreter.GetInputTensor(0)
		input.Float32s()[0] = float32(v)
		interpreter.Invoke()
		got := float64(interpreter.GetOutputTensor(0).Float32s()[0])
		want := math.Sin(v)
		if math.Abs(got-want) > 0.02 {
			log.Fatal("bad", i, v, math.Abs(got-want))
		}
		fmt.Printf("sin(%v) = %v\n", v, got)
	}
}
