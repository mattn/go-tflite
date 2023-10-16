package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"sort"

	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
)

func loadLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return labels, nil
}

func main() {
	var model_path, label_path, image_path string
	flag.StringVar(&model_path, "model", "mobilenet_quant_v1_224.tflite", "path to model file")
	flag.StringVar(&label_path, "label", "labels.txt", "path to label file")
	flag.StringVar(&image_path, "image", "peacock.png", "path to image file")
	flag.Parse()

	f, err := os.Open(image_path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	labels, err := loadLabels(label_path)
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	options.SetErrorReporter(func(msg string, user_data interface{}) {
		fmt.Println(msg)
	}, nil)
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Println("cannot create interpreter")
		return
	}
	defer interpreter.Delete()

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Println("allocate failed")
		return
	}

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	wanted_channels := input.Dim(3)
	wanted_type := input.Type()

	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	if wanted_type == tflite.UInt8 {
		bb := make([]byte, wanted_width*wanted_height*wanted_channels)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				bb[(y*dx+x)*3+0] = byte(float64(r) / 255.0)
				bb[(y*dx+x)*3+1] = byte(float64(g) / 255.0)
				bb[(y*dx+x)*3+2] = byte(float64(b) / 255.0)
			}
		}
		copy(input.UInt8s(), bb)
	} else if wanted_type == tflite.Float32 {
		ff := make([]float32, wanted_width*wanted_height*wanted_channels)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				ff[(y*dx+x)*3+0] = float32(r) / 65535.0
				ff[(y*dx+x)*3+1] = float32(g) / 65535.0
				ff[(y*dx+x)*3+2] = float32(b) / 65535.0
			}
		}
		copy(input.Float32s(), ff)
	} else {
		log.Println("is not wanted type")
		return
	}

	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Println("invoke failed")
		return
	}

	output := interpreter.GetOutputTensor(0)
	output_size := output.Dim(output.NumDims() - 1)
	b := make([]byte, output_size)
	type result struct {
		score float64
		index int
	}
	status = output.CopyToBuffer(&b[0])
	if status != tflite.OK {
		log.Println("output failed")
		return
	}
	results := []result{}
	for i := 0; i < output_size; i++ {
		score := float64(b[i]) / 255.0
		if score < 0.2 {
			continue
		}
		results = append(results, result{score: score, index: i})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	for i := 0; i < len(results); i++ {
		fmt.Printf("%02d: %s: %f\n", results[i].index, labels[results[i].index], results[i].score)
		if i > 5 {
			break
		}
	}
}
