package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"os/signal"
	"sort"
	"sync"
	"time"

	"github.com/mattn/go-tflite"
	//"github.com/mattn/go-tflite/delegates/edgetpu"
	"golang.org/x/image/colornames"

	"gocv.io/x/gocv"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "yolov3-tiny.tflite", "path to model file")
	labelPath = flag.String("label", "coco_labels.txt", "path to label file")
)

// headData holds one YOLO output head, dequantized to float32.
type headData struct {
	loc   []float32
	shape []int
}

type ssdResult struct {
	heads []headData
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
		return items[i].score > items[j].score
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
			if len(result) >= 20 {
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
		gocv.CvtColor(frame, &resized, gocv.ColorBGRToRGB)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if input.Type() == tflite.Float32 {
			resized.ConvertToWithParams(&resized, gocv.MatTypeCV32F, 1.0/255, 0)
			if ff, err := resized.DataPtrFloat32(); err == nil {
				copy(input.Float32s(), ff)
			}
		} else {
			if v, err := resized.DataPtrUint8(); err == nil {
				copy(input.UInt8s(), v)
			}
		}
		resized.Close()
		status := interpreter.Invoke()
		if status != tflite.OK {
			log.Println("invoke failed")
			return
		}

		var heads []headData
		for t := 0; t < interpreter.GetOutputTensorCount(); t++ {
			output := interpreter.GetOutputTensor(t)
			if output.NumDims() < 3 {
				continue
			}
			var loc []float32
			switch output.Type() {
			case tflite.UInt8:
				oqp := output.QuantizationParams()
				scale := oqp.Scale
				if scale == 0 {
					scale = 1
				}
				f := output.UInt8s()
				loc = make([]float32, len(f))
				for i, v := range f {
					loc[i] = float32(scale * float64(int(v)-oqp.ZeroPoint))
				}
			case tflite.Float32:
				loc = make([]float32, len(output.Float32s()))
				copy(loc, output.Float32s())
			}
			if loc != nil {
				heads = append(heads, headData{loc: loc, shape: output.Shape()})
			}
		}
		resultChan <- &ssdResult{
			heads: heads,
			mat:   frame,
		}
	}
}

type item struct {
	x1, y1, x2, y2 float32
	score          float32
	class          int
}

// Anchor sizes in model input pixels: 9 for full YOLOv3 (3 heads), 6 for
// YOLOv3-tiny (2 heads). Each head uses a group of 3, coarse heads first.
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

var anchorsTiny = []float32{
	10, 14,
	23, 27,
	37, 58,
	81, 82,
	135, 169,
	344, 319,
}

func sigmoid(v float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-v))))
}

// decodedHeads reports whether the model outputs already-decoded boxes, as
// YOLOv4 models converted with hunglc007/tensorflow-yolov4-tflite do: one
// [1,N,4] tensor of (cx,cy,w,h) in model input pixels and one [1,N,classes]
// tensor of per-class scores.
func decodedHeads(heads []headData) (*headData, *headData, bool) {
	if len(heads) != 2 {
		return nil, nil, false
	}
	b, s := &heads[0], &heads[1]
	if len(b.shape) != 3 || len(s.shape) != 3 {
		return nil, nil, false
	}
	if b.shape[2] != 4 {
		b, s = s, b
	}
	if b.shape[2] != 4 || s.shape[2] <= 4 || b.shape[1] != s.shape[1] {
		return nil, nil, false
	}
	return b, s, true
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

	/*
		devices, err := edgetpu.DeviceList()
		if err == nil && len(devices) > 0 {
			options.AddDelegate(edgetpu.New(devices[0]))
		}
	*/
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

		const scoreThreshold = 0.3

		var items []item
		size := result.mat.Size()
		if b, s, ok := decodedHeads(result.heads); ok {
			classes := s.shape[2]
			scaleX := float32(size[1]) / float32(wanted_width)
			scaleY := float32(size[0]) / float32(wanted_height)
			for i := 0; i < b.shape[1]; i++ {
				sc := s.loc[i*classes : (i+1)*classes]
				class := argmax(sc)
				score := sc[class]
				if score < scoreThreshold {
					continue
				}
				cx := b.loc[i*4+0] * scaleX
				cy := b.loc[i*4+1] * scaleY
				w := b.loc[i*4+2] * scaleX
				h := b.loc[i*4+3] * scaleY
				items = append(items, item{
					x1:    cx - w/2,
					y1:    cy - h/2,
					x2:    cx + w/2,
					y2:    cy + h/2,
					score: score,
					class: class,
				})
			}
		}
		anchorList := anchors
		if len(result.heads) == 2 {
			anchorList = anchorsTiny
		}
		for hi, head := range result.heads {
			shape := head.shape
			loc := head.loc
			if len(shape) < 4 {
				continue
			}

			// Heads come as either [1,h,w,anchors,5+classes] or with the
			// anchors folded into the channel dimension.
			var anc, per int
			if len(shape) == 5 {
				anc, per = shape[3], shape[4]
			} else if shape[3]%3 == 0 {
				anc, per = 3, shape[3]/3
			} else {
				anc, per = 1, shape[3]
			}

			// Coarse heads use the large anchors at the end of the list.
			maskStart := (len(result.heads) - 1 - hi) * 3
			if (maskStart+anc)*2 > len(anchorList) {
				maskStart = 0
			}

			gridH, gridW := shape[1], shape[2]
			strideX := float32(size[1]) / float32(gridW)
			strideY := float32(size[0]) / float32(gridH)
			scaleX := float32(size[1]) / float32(wanted_width)
			scaleY := float32(size[0]) / float32(wanted_height)
			for i := 0; i < gridH; i++ {
				for j := 0; j < gridW; j++ {
					for k := 0; k < anc; k++ {
						idx := ((i*gridW+j)*anc + k) * per
						objectness := sigmoid(loc[idx+4])
						if objectness < scoreThreshold {
							continue
						}
						class := argmax(loc[idx+5 : idx+per])
						score := objectness * sigmoid(loc[idx+5+class])
						if score < scoreThreshold {
							continue
						}
						// bx = (sigmoid(tx)+cell)*stride, w = anchor*exp(tw),
						// with anchors given in model input pixels.
						cx := (float32(j) + sigmoid(loc[idx+0])) * strideX
						cy := (float32(i) + sigmoid(loc[idx+1])) * strideY
						w := anchorList[(maskStart+k)*2+0] * float32(math.Exp(float64(loc[idx+2]))) * scaleX
						h := anchorList[(maskStart+k)*2+1] * float32(math.Exp(float64(loc[idx+3]))) * scaleY
						items = append(items, item{
							x1:    cx - w/2,
							y1:    cy - h/2,
							x2:    cx + w/2,
							y2:    cy + h/2,
							score: score,
							class: class,
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
