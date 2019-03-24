package main

import (
	"flag"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"log"
	"os"

	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
)

var colors = [21][3]uint8{
	{0, 0, 0},
	{128, 0, 0},
	{0, 128, 0},
	{128, 128, 0},
	{0, 0, 128},
	{128, 0, 128},
	{0, 128, 128},
	{128, 128, 128},
	{64, 0, 0},
	{192, 0, 0},
	{64, 128, 0},
	{192, 128, 0},
	{64, 0, 128},
	{192, 0, 128},
	{64, 128, 128},
	{192, 128, 128},
	{0, 64, 0},
	{128, 64, 0},
	{0, 192, 0},
	{128, 192, 0},
	{0, 64, 128},
}

func main() {
	var model_path, image_path string
	flag.StringVar(&model_path, "model", "deeplabv3_257_mv_gpu.tflite", "path to model file")
	flag.StringVar(&image_path, "image", "example.jpg", "path to image file")
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

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Fatal("allocate failed")
	}

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	wanted_type := input.Type()

	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()
	if wanted_type == tflite.Float32 {
		ff := make([]float32, wanted_width*wanted_height*3)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				ff[(y*wanted_width+x)*3+0] = ((float32(r) / 255) - 127.0) / 127.0
				ff[(y*wanted_width+x)*3+1] = ((float32(g) / 255) - 127.0) / 127.0
				ff[(y*wanted_width+x)*3+2] = ((float32(b) / 255) - 127.0) / 127.0
			}
		}
		copy(input.Float32s(), ff)
	} else {
		bb := make([]byte, wanted_width*wanted_height*3)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				bb[(y*dx+x)*3+0] = byte(b)
				bb[(y*dx+x)*3+1] = byte(g)
				bb[(y*dx+x)*3+2] = byte(r)
			}
		}
		copy(input.UInt8s(), bb)
	}

	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Fatal("invoke failed")
	}

	canvas := image.NewRGBA(resized.Bounds())

	output := interpreter.GetOutputTensor(0)
	ff := output.Float32s()

	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			ci := 0
			cv := float32(-32767)
			off := (y*dx + x) * 21
			for i := 0; i < 21; i++ {
				v := ff[off+i]
				if cv < v {
					cv = v
					ci = i
				}
			}
			c := colors[ci]
			canvas.Set(x, y, color.RGBA{R: c[0], G: c[1], B: c[2], A: 100})
		}
	}

	canvasImg := resize.Resize(uint(img.Bounds().Dx()), uint(img.Bounds().Dy()), canvas, resize.NearestNeighbor)
	base := image.NewRGBA(img.Bounds())
	draw.Draw(base, base.Bounds(), img, image.Pt(0, 0), draw.Src)
	draw.Draw(base, base.Bounds(), canvasImg, image.Pt(0, 0), draw.Over)

	out, err := os.Create("output.png")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	err = png.Encode(out, base)
	if err != nil {
		log.Fatal(err)
	}
}
