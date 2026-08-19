package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/mattn/go-tflite"
)

func main() {
	var model_path string
	flag.StringVar(&model_path, "model", "../../testdata/xor_model_sig.tflite", "path to model file")
	flag.Parse()

	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	if interpreter == nil {
		log.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()

	if interpreter.GetSignatureCount() == 0 {
		log.Fatal("model has no signatures")
	}
	for i := 0; i < interpreter.GetSignatureCount(); i++ {
		fmt.Printf("signature: %s\n", interpreter.GetSignatureKey(i))
	}

	runner := interpreter.GetSignatureRunner(interpreter.GetSignatureKey(0))
	if runner == nil {
		log.Fatal("cannot get signature runner")
	}
	defer runner.Delete()

	if runner.AllocateTensors() != tflite.OK {
		log.Fatal("allocate failed")
	}

	for i := 0; i < runner.GetInputCount(); i++ {
		fmt.Printf("  input: %s\n", runner.GetInputName(i))
	}
	for i := 0; i < runner.GetOutputCount(); i++ {
		fmt.Printf("  output: %s\n", runner.GetOutputName(i))
	}

	input := runner.GetInputTensor(runner.GetInputName(0))
	if input == nil {
		log.Fatal("cannot get input tensor")
	}

	for _, v := range [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}} {
		copy(input.Float32s(), v)
		if runner.Invoke() != tflite.OK {
			log.Fatal("invoke failed")
		}
		output := runner.GetOutputTensor(runner.GetOutputName(0))
		if output == nil {
			log.Fatal("cannot get output tensor")
		}
		fmt.Printf("%v xor %v = %v\n", v[0], v[1], int(output.Float32s()[0]+0.5))
	}
}
