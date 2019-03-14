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

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"github.com/faiface/pixel/text"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font/basicfont"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "mobilenet_quant_v1_224.tflite", "path to model file")
	labelPath = flag.String("label", "labels.txt", "path to label file")
)

type result struct {
	output []byte
	size   int
	img    image.Image
}

type class struct {
	score float64
	index int
}

func loadLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open("labels.txt")
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

func capture(wg *sync.WaitGroup, cam *gocv.VideoCapture, frameChan chan image.Image, ctx context.Context) {
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

		frameChan <- img
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

	// Start up the background capture
	frameChan := make(chan image.Image, 3)
	resultChan := make(chan result, 3)
	go capture(&wg, cam, frameChan, ctx)
	go detect(&wg, resultChan, frameChan, ctx)

	// Setup Pixel requirements for drawing boxes and labels
	mat := pixel.IM
	mat = mat.Moved(win.Bounds().Center())

	atlas := text.NewAtlas(basicfont.Face7x13, text.ASCII)
	imd := imdraw.New(nil)

	// Some local vars to calculate frame rate
	var (
		frames = 0
		second = time.Tick(time.Second)
	)

	for !win.Closed() {
		// Run inference if we have a new frame to read
		result := <-resultChan

		classes := make([]class, 0, result.size)
		b := result.output
		var i int
		for i = 0; i < result.size; i++ {
			score := float64(b[i]) / 255.0
			if score < 0.2 {
				continue
			}
			classes = append(classes, class{score: score, index: i})
		}
		sort.Slice(classes, func(i, j int) bool {
			return classes[i].score > classes[j].score
		})

		if len(classes) > 5 {
			classes = classes[:5]
		}
		pic := pixel.PictureDataFromImage(result.img)
		bounds := pic.Bounds()
		sprite := pixel.NewSprite(pic, bounds)

		win.Clear(colornames.Black)
		sprite.Draw(win, mat)

		for i, class := range classes {
			s := fmt.Sprintf("%d %.5f %s\n", i, class.score, labels[class.index])
			txt := text.New(pixel.V(10.0, 470.0-float64(30*i)), atlas)
			txt.Color = color.White
			txt.WriteString(s)
			txt.Draw(win, pixel.IM.Scaled(txt.Orig, 2))
		}

		imd.Draw(win)
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

func detect(wg *sync.WaitGroup, resultChan chan<- result, frameChan <-chan image.Image, ctx context.Context) {
	defer wg.Done()

	model := tflite.NewModelFromFile(*modelPath)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)

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
	//wanted_type := input.Type()

	bb := make([]byte, wanted_width*wanted_height*wanted_channels)

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
			input.CopyFromBuffer(bb)
			status = interpreter.Invoke()
			if status != tflite.OK {
				log.Fatal("invoke failed")
			}

			output := interpreter.GetOutputTensor(0)
			output_size := output.Dim(output.NumDims() - 1)
			b := make([]byte, output_size)
			status = output.CopyToBuffer(b)
			if status != tflite.OK {
				log.Fatal("output failed")
			}
			resultChan <- result{
				output: b,
				size:   output_size,
				img:    img,
			}

		default:
		}
	}
}

func main() {
	flag.Parse()
	pixelgl.Run(run)
}
