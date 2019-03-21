package main

import (
	"bufio"
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/mattn/go-tflite"
	"golang.org/x/image/colornames"

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"github.com/faiface/pixel/text"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
	"golang.org/x/image/font/basicfont"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "mobilenet_quant_v1_224.tflite", "path to model file")
	labelPath = flag.String("label", "labels.txt", "path to label file")
)

type quantResult struct {
	output []byte
	img    image.Image
}

func (r *quantResult) Image() image.Image {
	return r.img
}

type ssdResult struct {
	loc   [][4]float32
	clazz []float32
	score []float32
	img   image.Image
}

func (r *ssdResult) Image() image.Image {
	return r.img
}

type quantClass struct {
	score float64
	index int
}

type ssdClass struct {
	loc   [4]float32
	score float64
	index int
}

type result interface {
	Image() image.Image
}

func loadLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open(*labelPath)
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

func capture(wg *sync.WaitGroup, frameChan chan image.Image, ctx context.Context, cam *gocv.VideoCapture, win *pixelgl.Window) {
	defer wg.Done()

	frame := gocv.NewMat()
	defer frame.Close()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if ok := cam.Read(&frame); !ok {
			log.Fatal("failed reading cam")
		}

		// Encode Mat as a bmp (uncompressed)
		buf, err := gocv.IMEncode(".png", frame)
		if err != nil {
			log.Fatalf("Error encoding frame: %v", err)
		}

		// Push the frame to the channel
		img, _, err := image.Decode(bytes.NewReader(buf))
		if err != nil {
			continue
		}
		if len(frameChan) == cap(frameChan) {
			continue
		}
		bounds := img.Bounds()
		height := 500 * bounds.Dy() / bounds.Dx()
		win.SetBounds(pixel.R(0, 0, 500, float64(height)))

		frameChan <- resize.Resize(500, uint(height), img, resize.NearestNeighbor)
	}
}

func run() {
	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatal(err)
	}

	// Setup Pixel window
	cfg := pixelgl.WindowConfig{
		Title:  "Thinger",
		Bounds: pixel.R(0, 0, 500, 500),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(2)

	cam, err := gocv.OpenVideoCapture(*video)
	if err != nil {
		log.Fatal("failed reading cam", err)
	}
	defer cam.Close()

	model := tflite.NewModelFromFile(*modelPath)
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
	wanted_channels := input.Dim(3)

	// Start up the background capture
	frameChan := make(chan image.Image, 2)
	resultChan := make(chan result, 2)
	go capture(&wg, frameChan, ctx, cam, win)
	go detect(&wg, resultChan, frameChan, ctx, interpreter, wanted_width, wanted_height, wanted_channels)

	// Setup Pixel requirements for drawing boxes and labels
	atlas := text.NewAtlas(basicfont.Face7x13, text.ASCII)
	imd := imdraw.New(nil)

	// Some local vars to calculate frame rate
	var (
		frames = 0
		second = time.Tick(time.Second)
	)

	for !win.Closed() {
		// Run inference if we have a new frame to read
		resulti := <-resultChan

		pic := pixel.PictureDataFromImage(resulti.Image())
		bounds := pic.Bounds()
		sprite := pixel.NewSprite(pic, bounds)

		win.Clear(colornames.Black)
		mat := pixel.IM
		mat = mat.Moved(win.Bounds().Center())
		sprite.Draw(win, mat)

		switch t := resulti.(type) {
		case *ssdResult:
			classes := make([]ssdClass, 0, len(t.clazz))
			var i int
			for i = 0; i < len(t.clazz); i++ {
				idx := int(t.clazz[i] + 1)
				score := float64(t.score[i])
				if score < 0.4 {
					continue
				}
				classes = append(classes, ssdClass{loc: t.loc[i], score: score, index: idx})
			}
			sort.Slice(classes, func(i, j int) bool {
				return classes[i].score > classes[j].score
			})

			if len(classes) > 5 {
				classes = classes[:5]
			}
			imd.Clear()
			size := t.img.Bounds()
			for i, class := range classes {
				label := "unknown"
				if class.index < len(labels) {
					label = labels[class.index]
				}
				imd.Color = colornames.Map[colornames.Names[class.index]]
				pos := pixel.V(float64(size.Dx())*float64(class.loc[1]), float64(size.Dy())*float64(1-class.loc[0]))
				imd.Push(pos)
				imd.Push(pixel.V(float64(size.Dx())*float64(class.loc[3]), float64(size.Dy())*float64(1-class.loc[0])))
				imd.Push(pixel.V(float64(size.Dx())*float64(class.loc[3]), float64(size.Dy())*float64(1-class.loc[2])))
				imd.Push(pixel.V(float64(size.Dx())*float64(class.loc[1]), float64(size.Dy())*float64(1-class.loc[2])))
				imd.Push(pos)
				imd.Line(1)
				txt := text.New(pos, atlas)
				txt.Color = imd.Color
				txt.WriteString(fmt.Sprintf("%d %.5f %s\n", i, class.score, label))
				txt.Draw(win, pixel.IM.Scaled(txt.Orig, 1))
			}
			imd.Draw(win)
		case *quantResult:
			b := t.output
			classes := make([]quantClass, 0, len(b))
			var i int
			for i = 0; i < len(b); i++ {
				score := float64(b[i]) / 255.0
				if score < 0.2 {
					continue
				}
				classes = append(classes, quantClass{score: score, index: i})
			}
			sort.Slice(classes, func(i, j int) bool {
				return classes[i].score > classes[j].score
			})

			if len(classes) > 5 {
				classes = classes[:5]
			}
			h := win.Bounds().H()
			for i, class := range classes {
				label := "unknown"
				if class.index < len(labels) {
					label = labels[class.index]
				}
				s := fmt.Sprintf("%d %.5f %s\n", i, class.score, label)
				txt := text.New(pixel.V(10.0, h-10-float64(10*i)), atlas)
				txt.Color = color.White
				txt.WriteString(s)
				txt.Draw(win, pixel.IM.Scaled(txt.Orig, 1))
			}
		}

		win.Update()

		// calculate frame rate
		frames++
		select {
		case <-second:
			win.SetTitle(fmt.Sprintf("%s | FPS: %d", cfg.Title, frames))
			frames = 0
		default:
		}
	}

	cancel()
	<-frameChan
	<-resultChan
	wg.Wait()
	close(frameChan)
	close(resultChan)
}

func detect(wg *sync.WaitGroup, resultChan chan<- result, frameChan <-chan image.Image, ctx context.Context, interpreter *tflite.Interpreter, wanted_width, wanted_height, wanted_channels int) {
	defer wg.Done()

	bb := make([]byte, wanted_width*wanted_height*wanted_channels)
	input := interpreter.GetInputTensor(0)

	for {
		select {
		case <-ctx.Done():
			return
		case img := <-frameChan:
			resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.NearestNeighbor)
			for y := 0; y < wanted_height; y++ {
				for x := 0; x < wanted_width; x++ {
					r, g, b, _ := resized.At(x, y).RGBA()
					bb[(y*wanted_width+x)*3+0] = byte(float64(b) / 255.0)
					bb[(y*wanted_width+x)*3+1] = byte(float64(g) / 255.0)
					bb[(y*wanted_width+x)*3+2] = byte(float64(r) / 255.0)
				}
			}
			copy(input.UInt8s(), bb)
			status := interpreter.Invoke()
			if status != tflite.OK {
				log.Fatal("invoke failed")
			}

			output := interpreter.GetOutputTensor(0)
			if output.Type() == tflite.Float32 {
				var loc [10][4]float32
				var clazz [10]float32
				var score [10]float32
				var nums [1]float32
				output.CopyToBuffer(&loc[0])
				interpreter.GetOutputTensor(1).CopyToBuffer(&clazz[0])
				interpreter.GetOutputTensor(2).CopyToBuffer(&score[0])
				interpreter.GetOutputTensor(3).CopyToBuffer(&nums[0])
				num := int(nums[0])

				resultChan <- &ssdResult{
					loc:   loc[:num],
					clazz: clazz[:num],
					score: score[:num],
					img:   img,
				}
			} else {
				output_size := output.Dim(output.NumDims() - 1)
				b := make([]byte, output_size)
				copy(b, output.UInt8s())
				resultChan <- &quantResult{
					output: b,
					img:    img,
				}
			}

		default:
		}
	}
}

func main() {
	flag.Parse()
	pixelgl.Run(run)
}
