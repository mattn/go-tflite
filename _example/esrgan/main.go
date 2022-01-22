package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/jpeg"
	"log"
	"os"

	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
)

func main() {
	var model_path, image_path string
	flag.StringVar(&model_path, "model", "ESRGAN.tflite", "path to model file")
	flag.StringVar(&image_path, "image", "lr-1.jpg", "path to image file")
	flag.Parse()

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

	f, err := os.Open(image_path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)

	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	ff := input.Float32s()
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			col := resized.At(x, y)
			r, g, b, _ := col.RGBA()
			ff[(y*dx+x)*3+0] = float32(r)
			ff[(y*dx+x)*3+1] = float32(g)
			ff[(y*dx+x)*3+2] = float32(b)
		}
	}

	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Println("invoke failed")
		return
	}

	output := interpreter.GetOutputTensor(0)
	ff = output.Float32s()
	dx, dy = int(output.Dim(1)), int(output.Dim(2))
	canvas := image.NewRGBA(image.Rect(0, 0, dx, dy))
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			canvas.Set(x, y, color.RGBA{
				R: uint8(ff[(y*dx+x)*3+0] / 255),
				G: uint8(ff[(y*dx+x)*3+1] / 255),
				B: uint8(ff[(y*dx+x)*3+2] / 255),
				A: 255,
			})
		}
	}
	f, err = os.Create("output.jpg")
	if err != nil {
		log.Println("cannot create image file")
		return
	}
	defer f.Close()
	err = jpeg.Encode(f, canvas, nil)
	if err != nil {
		log.Println("cannot output image file")
		return
	}
}
