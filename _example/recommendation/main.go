package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/mattn/go-tflite"
)

type movie struct {
	Title  string   `json:"title"`
	ID     int      `json:"id"`
	Genres []string `json:"genres"`
	Count  int      `json:"count"`
}

func main() {
	var model_path, vocab_path string
	var topK int
	flag.StringVar(&model_path, "model", "recommendation.tflite", "path to model file")
	flag.StringVar(&vocab_path, "vocab", "sorted_movie_vocab.json", "path to movie vocab file")
	flag.IntVar(&topK, "n", 10, "number of recommendations")
	flag.Parse()

	f, err := os.Open(vocab_path)
	if err != nil {
		log.Fatal(err)
	}
	var movies []movie
	err = json.NewDecoder(f).Decode(&movies)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	byID := map[int]movie{}
	for _, m := range movies {
		byID[m.ID] = m
	}

	// Context movie IDs are given as arguments; the most popular movies from
	// the vocab are used when none are given.
	var context []int
	for _, arg := range flag.Args() {
		id, err := strconv.Atoi(arg)
		if err != nil {
			log.Fatalf("movie id must be a number: %v", err)
		}
		context = append(context, id)
	}
	if len(context) == 0 {
		for _, m := range movies[:3] {
			context = append(context, m.ID)
		}
	}

	fmt.Println("watched:")
	for _, id := range context {
		if m, ok := byID[id]; ok {
			fmt.Printf("  %d: %s\n", id, m.Title)
		} else {
			fmt.Printf("  %d: (unknown)\n", id)
		}
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

	// The context is a fixed-size int32 vector of movie IDs padded with 0.
	input := interpreter.GetInputTensor(0).Int32s()
	for i := range input {
		input[i] = 0
	}
	for i, id := range context {
		if i < len(input) {
			input[i] = int32(id)
		}
	}

	if interpreter.Invoke() != tflite.OK {
		log.Fatal("invoke failed")
	}

	// One output holds the candidate movie IDs, the other their scores; the
	// order differs between the RNN and CNN variants, so pick by type.
	var ids []int32
	var scores []float32
	for i := 0; i < interpreter.GetOutputTensorCount(); i++ {
		output := interpreter.GetOutputTensor(i)
		switch output.Type() {
		case tflite.Int32:
			ids = output.Int32s()
		case tflite.Float32:
			scores = output.Float32s()
		}
	}
	if ids == nil || scores == nil {
		log.Fatal("unexpected model outputs")
	}

	inContext := map[int]bool{}
	for _, id := range context {
		inContext[id] = true
	}

	fmt.Println("recommended:")
	shown := 0
	for i, id := range ids {
		m, ok := byID[int(id)]
		if !ok || inContext[int(id)] {
			continue
		}
		fmt.Printf("  %.3f %s\n", scores[i], m.Title)
		shown++
		if shown >= topK {
			break
		}
	}
}
