package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/mattn/go-tflite"
)

func loadLabels(filename string) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// yamnet_class_map.csv: index,mid,display_name
	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	labels := make([]string, 0, len(records))
	for i, record := range records {
		if i == 0 {
			continue // header
		}
		labels = append(labels, record[2])
	}
	return labels, nil
}

func main() {
	var model_path, label_path, wav_path string
	flag.StringVar(&model_path, "model", "yamnet.tflite", "path to model file")
	flag.StringVar(&label_path, "label", "yamnet_class_map.csv", "path to class map file")
	flag.StringVar(&wav_path, "wav", "miaow_16k.wav", "path to 16kHz mono WAV file")
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

	input := interpreter.GetInputTensor(0)
	window := 1
	for i := 0; i < input.NumDims(); i++ {
		window *= input.Dim(i)
	}

	// Slide the model window over the clip with 50% overlap and average the
	// scores, the way the YAMNet documentation suggests for longer clips.
	scores := make([]float64, len(labels))
	windows := 0
	buf := make([]float32, window)
	for off := 0; off == 0 || off+window <= len(samples); off += window / 2 {
		for i := range buf {
			buf[i] = 0
		}
		copy(buf, samples[off:min(off+window, len(samples))])
		copy(input.Float32s(), buf)
		if interpreter.Invoke() != tflite.OK {
			log.Fatal("invoke failed")
		}
		for i, v := range interpreter.GetOutputTensor(0).Float32s() {
			if i < len(scores) {
				scores[i] += float64(v)
			}
		}
		windows++
	}

	type result struct {
		label string
		score float64
	}
	results := make([]result, len(labels))
	for i := range scores {
		results[i] = result{label: labels[i], score: scores[i] / float64(windows)}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	fmt.Printf("%s (%.1fs, %d windows)\n", wav_path, float64(len(samples))/16000, windows)
	for _, r := range results[:5] {
		fmt.Printf("  %-20s %.3f\n", r.label, r.score)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
