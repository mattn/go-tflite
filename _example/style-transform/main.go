package main

import (
	"bytes"
	"errors"
	"flag"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"os"

	"github.com/disintegration/imaging"
	"github.com/mattn/go-tflite"
)

func predict(img image.Image, mpredict string) ([]float32, error) {
	model := tflite.NewModelFromFile(mpredict)
	if model == nil {
		return nil, errors.New("cannot load model")
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	interpreter.AllocateTensors()
	input := interpreter.GetInputTensor(0)
	output := interpreter.GetOutputTensor(0)

	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	wanted_channel := input.Dim(3)

	resized := imaging.Resize(img, wanted_width, wanted_height, imaging.Linear)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	in := input.Float32s()
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			in[(y*dx+x)*wanted_channel+0] = float32(r) / 65536
			in[(y*dx+x)*wanted_channel+1] = float32(g) / 65536
			in[(y*dx+x)*wanted_channel+2] = float32(b) / 65536
		}
	}

	interpreter.Invoke()

	dx = output.Dim(1)
	dx = output.Dim(2)
	wanted_channel = output.Dim(3)
	ff := make([]float32, dx*dy*wanted_channel)
	copy(ff, output.Float32s())
	return ff, nil
}

func loadImage(name string) (image.Image, error) {
	b, err := ioutil.ReadFile(name)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	return img, nil
}

func main() {
	var finput string
	var fstyle string
	var mpredict string
	var mstyle string
	var foutput string

	flag.StringVar(&finput, "input-image", "belfry.jpg", "input image")
	flag.StringVar(&fstyle, "style-image", "style23.jpg", "style image")
	flag.StringVar(&mpredict, "predict-model", "style_predict_quantized_256.tflite", "predict model")
	flag.StringVar(&mstyle, "style-model", "style_transfer_quantized_dynamic.tflite", "transfer model")
	flag.StringVar(&foutput, "output-image", "output.png", "output image")
	flag.Parse()

	img, err := loadImage(finput)
	if err != nil {
		log.Fatal(err)
	}
	style, err := loadImage(fstyle)
	if err != nil {
		log.Fatal(err)
	}
	in2, err := predict(style, mpredict)
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile(mstyle)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	defer interpreter.Delete()

	origbounds := img.Bounds()
	size := origbounds.Dx()
	if size < origbounds.Dy() {
		size = origbounds.Dy()
	}
	interpreter.ResizeInputTensor(0, []int32{1, int32(size), int32(size), 3})
	interpreter.ResizeInputTensor(1, []int32{1, 1, 1, 100})
	interpreter.AllocateTensors()

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	wanted_channel := input.Dim(3)

	resized := imaging.Resize(img, wanted_width, wanted_height, imaging.Linear)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	in1 := make([]float32, dx*dy*wanted_channel)
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			in1[(y*dx+x)*wanted_channel+0] = float32(r) / 65535
			in1[(y*dx+x)*wanted_channel+1] = float32(g) / 65535
			in1[(y*dx+x)*wanted_channel+2] = float32(b) / 65535
		}
	}

	copy(interpreter.GetInputTensor(0).Float32s(), in1)
	copy(interpreter.GetInputTensor(1).Float32s(), in2)

	interpreter.Invoke()

	output := interpreter.GetOutputTensor(0)
	out := output.Float32s()
	dx = output.Dim(1)
	dy = output.Dim(2)
	wanted_channel = output.Dim(3)
	canvas := image.NewRGBA(image.Rect(0, 0, dx, dy))
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			r := out[(y*dx+x)*wanted_channel+0] * 65536
			g := out[(y*dx+x)*wanted_channel+1] * 65536
			b := out[(y*dx+x)*wanted_channel+2] * 65536
			canvas.Set(x, y, color.RGBA64{R: uint16(r), G: uint16(g), B: uint16(b), A: 65535})
		}
	}
	resized = imaging.Resize(canvas, origbounds.Dx(), origbounds.Dy(), imaging.Lanczos)
	f, err := os.Create(foutput)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	png.Encode(f, resized)
}
