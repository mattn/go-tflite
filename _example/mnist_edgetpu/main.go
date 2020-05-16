package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"os"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/edgetpu"
	"github.com/nfnt/resize"
)

func top(a []float32) int {
	t := 0
	m := float32(0)
	for i, e := range a {
		if i == 0 || e > m {
			m = e
			t = i
		}
	}
	return t
}

func main() {
	var verbosity int
	flag.IntVar(&verbosity, "verbosity", 0, "Edge TPU Verbosity")

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile("mnist_model.tflite")
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	devices, err := edgetpu.DeviceList()
	if err != nil {
		log.Printf("Could not get EdgeTPU devices: %v", err)
		return
	}
	if len(devices) == 0 {
		log.Println("No edge TPU devices found")
		return
	}

	edgetpuVersion, err := edgetpu.Version()
	if err != nil {
		log.Printf("Could not get EdgeTPU version: %v", err)
		return
	}
	fmt.Printf("EdgeTPU Version: %s\n", edgetpuVersion)
	edgetpu.Verbosity(verbosity)
	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	options.AddDelegate(edgetpu.New(devices[0]))
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	defer interpreter.Delete()

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Println("allocate failed")
		return
	}

	input := interpreter.GetInputTensor(0)
	resized := resize.Resize(28, 28, img, resize.NearestNeighbor)
	in := input.Float32s()
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			in[y*28+x] = (float32(b) + float32(g) + float32(r)) / 3.0 / 65535.0
		}
	}
	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Println("invoke failed")
		return
	}

	output := interpreter.GetOutputTensor(0)
	out := output.Float32s()
	fmt.Println(top(out))
}
