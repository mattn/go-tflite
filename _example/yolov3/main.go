package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"sort"
	"sync"
	"time"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/edgetpu"
	"golang.org/x/image/colornames"

	"gocv.io/x/gocv"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "yolov3-tiny.tflite", "path to model file")
	labelPath = flag.String("label", "coco_labels.txt", "path to label file")
)

type ssdResult struct {
	loc   []float32
	shape []int
	anc   int
	cls   int
	thr   float32
	mat   gocv.Mat
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

func calcIntersectionOverUnion(f1, f2 item) float32 {
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

func omitItems(items []item) []item {
	var result []item

	sort.Slice(items, func(i, j int) bool {
		return items[i].score < items[j].score
	})

	for _, f1 := range items {
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

func detect(ctx context.Context, wg *sync.WaitGroup, resultChan chan<- *ssdResult, interpreter *tflite.Interpreter, wanted_width, wanted_height, wanted_channels int, cam *gocv.VideoCapture) {
	defer wg.Done()
	defer close(resultChan)

	input := interpreter.GetInputTensor(0)
	output := interpreter.GetOutputTensor(0)
	typ := output.Type()
	shape := output.Shape()
	anc := 1
	if len(shape) == 5 {
		anc = shape[3]
	}

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
				//ff[i] = ff[i] / 255.0
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

		var loc []float32
		switch typ {
		case tflite.UInt8:
			f := output.UInt8s()
			loc = make([]float32, len(f), len(f))
			for i, v := range f {
				loc[i] = float32(v) / 255
			}
			resultChan <- &ssdResult{
				loc:   loc,
				thr:   0.8,
				anc:   anc,
				cls:   10,
				shape: shape,
				mat:   frame,
			}
		case tflite.Float32:
			f := output.Float32s()
			loc = make([]float32, len(f), len(f))
			for i, v := range f {
				loc[i] = v
			}
			resultChan <- &ssdResult{
				loc:   loc,
				thr:   0.3,
				anc:   anc,
				cls:   80,
				shape: shape,
				mat:   frame,
			}
		default:
			resultChan <- &ssdResult{
				mat: frame,
			}
		}
	}
}

type item struct {
	x1, y1, x2, y2 float32
	score          float32
	class          int
}

var anchors = []float32{
	10, 13,
	16, 30,
	33, 23,
	30, 61,
	62, 45,
	59, 119,
	116, 90,
	156, 198,
	373, 326,
}

func argmax(f []float32) int {
	r, m := 0, f[0]
	for i, v := range f {
		if v > m {
			m = v
			r = i
		}
	}
	return r
}

func main() {
	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatal(err)
	}
	_ = labels

	// Setup Pixel window
	window := gocv.NewWindow("Webcam Window")
	defer window.Close()

	ctx, cancel := context.WithCancel(context.Background())

	cam, err := gocv.OpenVideoCapture(*video)
	if err != nil {
		log.Printf("cannot open camera: %v", err)
		return
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

	devices, err := edgetpu.DeviceList()
	if err == nil && len(devices) > 0 {
		options.AddDelegate(edgetpu.New(devices[0]))
	}
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
		log.Print("allocate failed")
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

	// Some local vars to calculate frame rate
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

		var items []item
		loc := result.loc
		size := result.mat.Size()
		if len(loc) != 0 {
			shape := result.shape
			sx := float32(size[1]) / float32(shape[1])
			sy := float32(size[0]) / float32(shape[2])
			for i := 0; i < shape[1]; i++ {
				for j := 0; j < shape[2]; j++ {
					for k := 0; k < result.anc; k++ {
						idx := ((i*shape[2]+j)*result.anc + k) * shape[len(shape)-1]
						if loc[idx+4] < result.thr {
							continue
						}
						dx := anchors[k*2+0]
						dy := anchors[k*2+1]
						x1 := sx*float32(j) + sx*loc[idx+0]
						y1 := sy*float32(i) + sy*loc[idx+1]
						w := sx * float32(math.Log(float64(dx*float32(math.Exp(float64(loc[idx+2]))))))
						h := sy * float32(math.Log(float64(dy*float32(math.Exp(float64(loc[idx+3]))))))
						items = append(items, item{
							x1:    x1 - w/2,
							y1:    y1 - h/2,
							x2:    x1 + w/2,
							y2:    y1 + h/2,
							score: loc[idx+4],
							class: argmax(loc[idx+5 : idx+5+result.cls]),
						})
					}
				}
			}
		}

		items = omitItems(items)
		for i, item := range items {
			ci := item.class % len(colornames.Names)
			c := colornames.Map[colornames.Names[ci]]
			gocv.Rectangle(&result.mat, image.Rect(
				int(item.x1),
				int(item.y1),
				int(item.x2),
				int(item.y2),
			), c, 2)
			label := "unknown"
			if item.class < len(labels) {
				label = labels[item.class]
			}
			text := fmt.Sprintf("%d %s", i, label)
			gocv.PutText(&result.mat, text, image.Pt(
				int(item.x1),
				int(item.y1),
			), gocv.FontHersheySimplex, 0.5, c, 1)
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
