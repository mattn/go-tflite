package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"os"
	"sort"

	"golang.org/x/image/colornames"

	"github.com/llgcode/draw2d"
	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/mattn/go-tflite"

	"github.com/nfnt/resize"
)

var (
	inputPath  = flag.String("input", "example.jpg", "path to the input image file")
	outputPath = flag.String("output", "output.png", "path to the output image file")
	modelPath  = flag.String("model", "detect.tflite", "path to model file")
	labelPath  = flag.String("label", "labelmap.txt", "path to label file")
)

type ssdClass struct {
	loc   [4]float32
	score float64
	index int
}

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

func maxFloat32(f []float32) (int, float32) {
	mi := 0
	mf := float32(0)
	for i := 1; i < len(f); i++ {
		if mf < f[i] {
			mi = i
			mf = f[i]
		}
	}
	return mi, mf
}

func main() {
	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.Open(*inputPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	model := tflite.NewModelFromFile(*modelPath)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	options.SetErrorReporter(func(format string, v interface{}) {
		println(format, v)
	}, nil)
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
	wanted_channels := input.Dim(3)

	qp := input.QuantizationParams()
	log.Printf("width: %v, height: %v, type: %v, scale: %v, zeropoint: %v", wanted_width, wanted_height, input.Type(), qp.Scale, qp.ZeroPoint)
	log.Printf("input tensor count: %v, output tensor count: %v", interpreter.GetInputTensorCount(), interpreter.GetOutputTensorCount())
	if qp.Scale == 0 {
		qp.Scale = 1
	}
	bb := make([]byte, wanted_width*wanted_height*wanted_channels)

	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.NearestNeighbor)
	for y := 0; y < wanted_height; y++ {
		for x := 0; x < wanted_width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			bb[(y*wanted_width+x)*3+0] = byte(float64(int(b)-qp.ZeroPoint) * qp.Scale)
			bb[(y*wanted_width+x)*3+1] = byte(float64(int(g)-qp.ZeroPoint) * qp.Scale)
			bb[(y*wanted_width+x)*3+2] = byte(float64(int(r)-qp.ZeroPoint) * qp.Scale)
		}
	}
	copy(input.UInt8s(), bb)

	status = interpreter.Invoke()
	if status != tflite.OK {
		log.Println("invoke failed")
		return
	}

	output := interpreter.GetOutputTensor(0)

	var loc [10][4]float32
	var clazz [10]float32
	var score [10]float32
	var nums [1]float32
	output.CopyToBuffer(&loc[0])
	interpreter.GetOutputTensor(1).CopyToBuffer(&clazz[0])
	interpreter.GetOutputTensor(2).CopyToBuffer(&score[0])
	interpreter.GetOutputTensor(3).CopyToBuffer(&nums[0])
	num := int(nums[0])

	canvas := image.NewRGBA(img.Bounds())
	gc := draw2dimg.NewGraphicContext(canvas)
	draw2d.SetFontFolder("C:/Windows/fonts")
	draw2d.SetFontNamer(func(fontData draw2d.FontData) string {
		return "MSGothic.ttc"
	})
	gc.DrawImage(img)

	classes := make([]ssdClass, 0, len(clazz))
	var i int
	for i = 0; i < num; i++ {
		idx := int(clazz[i] + 1)
		score := float64(score[i])
		if score < 0.5 {
			continue
		}
		if loc[i][2]-loc[i][0] > 0.8 || loc[i][3]-loc[i][1] > 0.6 {
			continue
		}
		if loc[i][2]-loc[i][0] < 0.1 || loc[i][3]-loc[i][1] < 0.1 {
			continue
		}
		classes = append(classes, ssdClass{loc: loc[i], score: score, index: idx})
	}
	sort.Slice(classes, func(i, j int) bool {
		return classes[i].score > classes[j].score
	})

	if len(classes) > 5 {
		classes = classes[:5]
	}
	size := img.Bounds()
	for i, class := range classes {
		label := "unknown"
		if class.index < len(labels) {
			label = labels[class.index]
		}
		gc.BeginPath()
		gc.SetStrokeColor(colornames.Map[colornames.Names[class.index]])
		gc.SetLineWidth(1)
		gc.MoveTo(float64(size.Dx())*float64(class.loc[1]), float64(size.Dy())*float64(class.loc[0]))
		gc.LineTo(float64(size.Dx())*float64(class.loc[3]), float64(size.Dy())*float64(class.loc[0]))
		gc.LineTo(float64(size.Dx())*float64(class.loc[3]), float64(size.Dy())*float64(class.loc[2]))
		gc.LineTo(float64(size.Dx())*float64(class.loc[1]), float64(size.Dy())*float64(class.loc[2]))
		gc.Close()
		gc.Stroke()
		s := fmt.Sprintf("%d %.5f %s\n", i, class.score, label)
		log.Println(s)
		gc.StrokeStringAt(s, float64(size.Dx())*float64(class.loc[1]), float64(size.Dy())*float64(class.loc[0]))
	}
	draw2dimg.SaveToPngFile(*outputPath, canvas)
}
