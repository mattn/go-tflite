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
	flag.StringVar(&model_path, "model", "model.tflite", "path to model file")
	flag.StringVar(&label_path, "label", "labels.txt", "path to label file")
	flag.StringVar(&image_path, "image", "", "path to image file of a hand gesture")
	flag.Parse()

	if image_path == "" {
		flag.Usage()
		os.Exit(1)
	}

	f, err := os.Open(image_path)
	if err != nil {
		log.Fatal(err)
	}
	img, _, err := image.Decode(f)
	f.Close()
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
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()

	if interpreter.AllocateTensors() != tflite.OK {
		log.Fatal("allocate failed")
	}

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)

	// MobileNet-style preprocessing: RGB scaled to [-1, 1].
	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.Bilinear)
	in := input.Float32s()
	for y := 0; y < wanted_height; y++ {
		for x := 0; x < wanted_width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			in[(y*wanted_width+x)*3+0] = float32(r>>8)/127.5 - 1
			in[(y*wanted_width+x)*3+1] = float32(g>>8)/127.5 - 1
			in[(y*wanted_width+x)*3+2] = float32(b>>8)/127.5 - 1
		}
	}

	if interpreter.Invoke() != tflite.OK {
		log.Fatal("invoke failed")
	}

	type result struct {
		label string
		score float32
	}
	var results []result
	for i, v := range interpreter.GetOutputTensor(0).Float32s() {
		if i < len(labels) {
			results = append(results, result{label: labels[i], score: v})
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	for _, r := range results[:3] {
		fmt.Printf("%-12s %.3f\n", r.label, r.score)
	}
}
