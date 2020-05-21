package main

import (
	"bytes"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"net/http"

	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
	"github.com/vincent-petithory/dataurl"
)

func argmax(a []float32) int {
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
	model := tflite.NewModelFromFile("mnist_model.tflite")
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Println("allocate failed")
		return
	}

	input := interpreter.GetInputTensor(0)
	output := interpreter.GetOutputTensor(0)

	http.HandleFunc("/picture", func(w http.ResponseWriter, r *http.Request) {
		dataURL, err := dataurl.Decode(r.Body)
		defer r.Body.Close()
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if dataURL.ContentType() != "image/png" {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		img, _, err := image.Decode(bytes.NewReader(dataURL.Data))
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
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
		fmt.Fprintf(w, "%v", argmax(output.Float32s()))
	})

	http.Handle("/", http.FileServer(http.Dir("static")))
	http.ListenAndServe(":8080", nil)
}
