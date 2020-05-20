package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"sort"
	"sync"
	"time"

	"github.com/mattn/go-tflite"

	"gocv.io/x/gocv"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "face_detection_front.tflite", "path to model file")
	labelPath = flag.String("label", "labelmap.txt", "path to label file")
)

type ssdResult struct {
	loc   []float32
	score []float32
	mat   gocv.Mat
}

func copySlice(f []float32) []float32 {
	ff := make([]float32, len(f), len(f))
	copy(ff, f)
	return ff
}

func detect(ctx context.Context, wg *sync.WaitGroup, resultChan chan<- *ssdResult, interpreter *tflite.Interpreter, wanted_width, wanted_height, wanted_channels int, cam *gocv.VideoCapture) {
	defer wg.Done()
	defer close(resultChan)

	input := interpreter.GetInputTensor(0)
	qp := input.QuantizationParams()
	log.Printf("width: %v, height: %v, type: %v, scale: %v, zeropoint: %v", wanted_width, wanted_height, input.Type(), qp.Scale, qp.ZeroPoint)
	log.Printf("input tensor count: %v, output tensor count: %v", interpreter.GetInputTensorCount(), interpreter.GetOutputTensorCount())

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if len(resultChan) == cap(resultChan) {
			continue
		}

		frame := gocv.NewMat()
		if ok := cam.Read(&frame); !ok {
			frame.Close()
			break
		}

		resized := gocv.NewMat()
		if input.Type() == tflite.Float32 {
			frame.ConvertTo(&resized, gocv.MatTypeCV32F)
			gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
			ff, err := resized.DataPtrFloat32()
			if err != nil {
				fmt.Println(err)
				continue
			}
			for i := 0; i < len(ff); i++ {
				ff[i] = (ff[i] - 127.5) / 127.5
			}
			copy(input.Float32s(), ff)
		} else {
			gocv.Resize(frame, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
			copy(input.UInt8s(), resized.DataPtrUint8())
		}
		resized.Close()
		status := interpreter.Invoke()
		if status != tflite.OK {
			log.Println("invoke failed")
			return
		}

		resultChan <- &ssdResult{
			loc:   copySlice(interpreter.GetOutputTensor(0).Float32s()),
			score: copySlice(interpreter.GetOutputTensor(1).Float32s()),
			mat:   frame,
		}
	}
}

type point struct {
	x, y float32
}

type face struct {
	x1, y1, x2, y2 float32
	score          float32
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func calcIntersectionOverUnion(f1, f2 face) float32 {
	xmin1 := min(f1.x1, f1.x2)
	ymin1 := min(f1.y1, f1.y2)
	xmax1 := max(f1.x1, f1.x2)
	ymax1 := max(f1.y1, f1.y2)
	xmin2 := min(f2.x1, f2.x2)
	ymin2 := min(f2.y1, f2.y2)
	xmax2 := max(f2.x1, f2.x2)
	ymax2 := max(f2.y1, f2.y2)

	area1 := (ymax1 - ymin1) * (xmax1 - xmin1)
	area2 := (ymax2 - ymin2) * (xmax2 - xmin2)
	if area1 <= 0 || area2 <= 0 {
		return 0.0
	}

	ixmin := max(xmin1, xmin2)
	iymin := max(ymin1, ymin2)
	ixmax := min(xmax1, xmax2)
	iymax := min(ymax1, ymax2)

	iarea := max(iymax-iymin, 0.0) * max(ixmax-ixmin, 0.0)

	return iarea / (area1 + area2 - iarea)
}

func omitFaces(faces []face) []face {
	var result []face

	sort.Slice(faces, func(i, j int) bool {
		return faces[i].score < faces[j].score
	})

	for _, f1 := range faces {
		ignore := false
		for _, f2 := range result {
			iou := calcIntersectionOverUnion(f1, f2)
			if iou >= 0.3 {
				ignore = true
				break
			}
		}

		if !ignore {
			result = append(result, f1)
			if len(result) > 20 {
				break
			}
		}
	}
	return result
}

func main() {
	flag.Parse()

	// Setup Pixel window
	window := gocv.NewWindow("Webcam Window")
	defer window.Close()

	ctx, cancel := context.WithCancel(context.Background())

	cam, err := gocv.OpenVideoCapture(*video)
	if err != nil {
		log.Fatal("failed reading cam", err)
	}
	defer cam.Close()

	window.ResizeWindow(
		int(cam.Get(gocv.VideoCaptureFrameWidth)),
		int(cam.Get(gocv.VideoCaptureFrameHeight)),
	)

	model := tflite.NewModelFromFile(*modelPath)
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
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

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	wanted_channels := input.Dim(3)

	var wg sync.WaitGroup
	wg.Add(1)

	// Start up the background capture
	resultChan := make(chan *ssdResult, 1)
	go detect(ctx, &wg, resultChan, interpreter, wanted_width, wanted_height, wanted_channels, cam)

	sc := make(chan os.Signal, 1)
	defer close(sc)
	signal.Notify(sc, os.Interrupt)
	go func() {
		<-sc
		cancel()
	}()

	strides := []int{8, 16}
	anchors := []int{2, 6}

	var vanchors []point
	for i := 0; i < 2; i++ {
		stride := strides[i]
		gridCols := (wanted_width + stride - 1) / stride
		gridRows := (wanted_height + stride - 1) / stride
		anchorNum := anchors[i]

		var anchor point
		for gridY := 0; gridY < gridRows; gridY++ {
			anchor.y = float32(stride) * (float32(gridY) + 0.5)
			for gridX := 0; gridX < gridCols; gridX++ {
				anchor.x = float32(stride) * (float32(gridX) + 0.5)
				for n := 0; n < anchorNum; n++ {
					vanchors = append(vanchors, anchor)
				}
			}
		}
	}

	var (
		frames = 0
		second = time.Tick(time.Second)
	)

	for {
		// Run inference if we have a new frame to read
		result, ok := <-resultChan
		if !ok {
			break
		}

		size := result.mat.Size()

		var faces []face
		for i, _ := range vanchors {
			score0 := result.score[i]
			score := 1.0 / (1.0 + math.Exp(float64(-score0)))

			if score < 0.75 {
				continue
			}
			p := result.loc[i*16 : i*16+4]

			sx := p[0]
			sy := p[1]
			w := p[2]
			h := p[3]
			anchor := vanchors[i]
			cx := sx + float32(anchor.x)
			cy := sy + float32(anchor.y)
			cx /= float32(wanted_width)
			cy /= float32(wanted_height)
			w /= float32(wanted_width)
			h /= float32(wanted_height)

			faces = append(faces, face{
				x1: (cx - w*0.5) * float32(size[1]),
				y1: (cy - h*0.5) * float32(size[0]),
				x2: (cx + w*0.5) * float32(size[1]),
				y2: (cy + h*0.5) * float32(size[0]),
			})
		}
		faces = omitFaces(faces)
		fmt.Println(len(faces))

		for _, face := range faces {
			gocv.Rectangle(&result.mat, image.Rect(
				int(face.x1),
				int(face.y1),
				int(face.x2),
				int(face.y2),
			), color.RGBA{R: 255, G: 0, B: 0}, 2)
		}

		window.IMShow(result.mat)
		result.mat.Close()

		k := window.WaitKey(1)
		if k == 0x1b {
			break
		}
		if window.GetWindowProperty(gocv.WindowPropertyVisible) == 0 {
			break
		}

		// calculate frame rate
		frames++
		select {
		case <-second:
			window.SetWindowTitle(fmt.Sprintf("SSD | FPS: %d", frames))
			frames = 0
		default:
		}
	}

	cancel()
	wg.Wait()
}
