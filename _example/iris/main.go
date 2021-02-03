package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/mattn/go-tflite"
)

func loadData() ([][]float32, []string, error) {
	f, err := os.Open("iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	var resultV [][]float32
	var resultS []string

	scanner := bufio.NewScanner(f)
	// skip header
	scanner.Scan()
	for scanner.Scan() {
		var f1, f2, f3, f4 float32
		var s string
		n, err := fmt.Sscanf(scanner.Text(), "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, &s)
		if n != 5 || err != nil {
			continue
		}
		resultV = append(resultV, []float32{f1, f2, f3, f4})
		resultS = append(resultS, s)
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, err
	}
	return resultV, resultS, nil
}

func main() {
	xx, yy, err := loadData()
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile("iris.tflite")
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	ss := []string{
		"Iris-setosa",
		"Iris-versicolor",
		"Iris-virginica",
	}
	a := 0
	for i := 0; i < len(xx); i++ {
		interpreter.GetInputTensor(0).CopyFromBuffer(xx[i])
		interpreter.Invoke()
		v := interpreter.GetOutputTensor(0).Float32s()
		var mv float32
		var mj int
		for j, vv := range v {
			if mv < vv {
				mv = vv
				mj = j
			}
		}
		if ss[mj] == yy[i] {
			a++
		}
	}
	fmt.Printf("%.02f\n", float64(a)/float64(len(xx))*100)
}
