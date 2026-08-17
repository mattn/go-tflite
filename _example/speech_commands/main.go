package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/mattn/go-tflite"
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
	var model_path, label_path, wav_path string
	flag.StringVar(&model_path, "model", "conv_actions_frozen.tflite", "path to model file")
	flag.StringVar(&label_path, "label", "conv_actions_labels.txt", "path to label file")
	flag.StringVar(&wav_path, "wav", "yes_1000ms.wav", "path to 16kHz mono WAV file")
	flag.Parse()

	labels, err := loadLabels(label_path)
	if err != nil {
		log.Fatal(err)
	}

	samples, rate, err := readWAV(wav_path)
	if err != nil {
		log.Fatal(err)
	}
	if rate != 16000 {
		log.Fatalf("%s: sample rate must be 16kHz but %dHz", wav_path, rate)
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

	// The model takes two inputs: the waveform as [16000,1] float32 and the
	// sample rate as a scalar int32.
	for i := 0; i < interpreter.GetInputTensorCount(); i++ {
		input := interpreter.GetInputTensor(i)
		switch input.Type() {
		case tflite.Float32:
			buf := input.Float32s()
			for j := range buf {
				buf[j] = 0
			}
			copy(buf, samples)
		case tflite.Int32:
			input.Int32s()[0] = int32(rate)
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
		fmt.Printf("%-10s %.3f\n", r.label, r.score)
	}
}
