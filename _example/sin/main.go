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
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	for i := -180; i < 180; i++ {
		v := float32(i) * math.Pi / 180.0
		input := interpreter.GetInputTensor(0)
		input.SetFloat32s([]float32{v})
		interpreter.Invoke()
		output := interpreter.GetOutputTensor(0)
		got := float64(output.Float32s()[0])
		want := math.Sin(float64(v))
		if math.Abs(got-want) > 0.02 {
			log.Println("bad", i, v, math.Abs(got-want))
			return
		}
		fmt.Printf("sin(%v) = %v\n", v, got)
	}
}
