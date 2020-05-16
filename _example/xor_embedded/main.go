package main

//go:generate statik

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"

	"github.com/mattn/go-tflite"
	_ "github.com/mattn/go-tflite/_example/xor_embedded/statik"
	"github.com/rakyll/statik/fs"
)

func main() {
	statikFS, err := fs.New()
	if err != nil {
		log.Fatal(err)
	}
	f, err := statikFS.Open("/xor_model.tflite")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	b, err := ioutil.ReadAll(f)
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModel(b)
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
