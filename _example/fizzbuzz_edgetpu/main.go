package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/edgetpu"
)

func bin(n int, num_digits int) []uint8 {
	f := make([]byte, num_digits)
	for i := uint(0); i < uint(num_digits); i++ {
		f[i] = uint8((n >> i) & 1) * 255
	}
	return f[:]
}

func dec(b []uint8) int {
	for i := 0; i < len(b); i++ {
		if b[i] > 100 {
			return i
		}
	}
	panic("Sorry, I'm wrong")
}

func display(v []uint8, i int) {
	switch dec(v) {
	case 0:
		fmt.Println(i)
	case 1:
		fmt.Println("Fizz")
	case 2:
		fmt.Println("Buzz")
	case 3:
		fmt.Println("FizzBuzz")
	}
}

func main() {
	var verbosity int
	flag.IntVar(&verbosity, "verbosity", 0, "Edge TPU Verbosity")
	flag.Parse()

	model := tflite.NewModelFromFile("fizzbuzz_model_quant_edgetpu.tflite")
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	devices, err := edgetpu.DeviceList()
	if err != nil {
		log.Fatalf("Could not get EdgeTPU devices: %v", err)
	}
	if len(devices) == 0 {
		log.Fatal("No edge TPU devices found")
	}

	edgetpuVersion, err := edgetpu.Version()
	if err != nil {
		log.Fatalf("Could not get EdgeTPU version: %v", err)
	}
	fmt.Printf("EdgeTPU Version: %s\n", edgetpuVersion)

	edgetpu.Verbosity(verbosity)
	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	options.AddDelegate(edgetpu.New(devices[0]))
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	for i := 1; i <= 100; i++ {
		buf := bin(i, 7)
		copy(interpreter.GetInputTensor(0).UInt8s(), buf)
		interpreter.Invoke()
		display(interpreter.GetOutputTensor(0).UInt8s(), i)
	}
}
