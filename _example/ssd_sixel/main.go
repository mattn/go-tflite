package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"os/signal"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mattn/go-sixel"
	"github.com/mattn/go-tflite"

	"gocv.io/x/gocv"
	"golang.org/x/image/colornames"
)

var (
	video     = flag.String("camera", "0", "video cature")
	modelPath = flag.String("model", "detect.tflite", "path to model file")
	labelPath = flag.String("label", "labelmap.txt", "path to label file")
	limits    = flag.Int("limits", 5, "limits of items")
)

type ssdResult struct {
	loc   []float32
	clazz []float32
	score []float32
}

type ssdClass struct {
	loc   []float32
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

func copySlice(f []float32) []float32 {
	ff := make([]float32, len(f), len(f))
	copy(ff, f)
	return ff
}

// cubePalette is a fixed 6x6x6 color cube. Setting it as the encoder's
// fixed palette skips the per-frame adaptive quantization that otherwise
// dominates frame time.
var cubePalette = func() color.Palette {
	p := make(color.Palette, 0, 216)
	for r := 0; r < 6; r++ {
		for g := 0; g < 6; g++ {
			for b := 0; b < 6; b++ {
				p = append(p, color.RGBA{uint8(r * 51), uint8(g * 51), uint8(b * 51), 255})
			}
		}
	}
	return p
}()

// capture reads frames from cam paced to the source frame rate and feeds
// them to the display loop. Each frame is also offered to the detector, but
// only when it is idle: display never waits for inference, so playback stays
// smooth even when Invoke is slower than the frame interval.
func capture(ctx context.Context, wg *sync.WaitGroup, frameChan chan<- gocv.Mat, detectChan chan<- gocv.Mat, cam *gocv.VideoCapture) {
	defer wg.Done()
	defer close(frameChan)
	defer close(detectChan)

	// Pace file playback to the source frame rate; without this the whole
	// pipeline runs as fast as it can and video files play too fast. For
	// cameras the blocking Read paces us at the same rate, so the ticker
	// costs nothing.
	var tick <-chan time.Time
	if fps := cam.Get(gocv.VideoCaptureFPS); fps > 0 {
		ticker := time.NewTicker(time.Duration(float64(time.Second) / fps))
		defer ticker.Stop()
		tick = ticker.C
	}

	for {
		if tick != nil {
			select {
			case <-ctx.Done():
				return
			case <-tick:
			}
		} else {
			select {
			case <-ctx.Done():
				return
			default:
			}
		}

		frame := gocv.NewMat()
		if ok := cam.Read(&frame); !ok {
			frame.Close()
			return
		}

		clone := frame.Clone()
		select {
		case detectChan <- clone:
		default:
			// The detector is still busy with an earlier frame; skip this one.
			clone.Close()
		}

		select {
		case <-ctx.Done():
			frame.Close()
			return
		case frameChan <- frame:
		}
	}
}

// detect runs inference on the frames it receives and publishes the latest
// result for the display loop to pick up.
func detect(ctx context.Context, wg *sync.WaitGroup, detectChan <-chan gocv.Mat, latest *atomic.Pointer[ssdResult], interpreter *tflite.Interpreter, wanted_width, wanted_height int) {
	defer wg.Done()

	input := interpreter.GetInputTensor(0)
	qp := input.QuantizationParams()
	log.Printf("width: %v, height: %v, type: %v, scale: %v, zeropoint: %v", wanted_width, wanted_height, input.Type(), qp.Scale, qp.ZeroPoint)
	log.Printf("input tensor count: %v, output tensor count: %v", interpreter.GetInputTensorCount(), interpreter.GetOutputTensorCount())
	if qp.Scale == 0 {
		qp.Scale = 1
	}

	for frame := range detectChan {
		resized := gocv.NewMat()
		gocv.CvtColor(frame, &resized, gocv.ColorBGRToRGB)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if input.Type() == tflite.Float32 {
			resized.ConvertToWithParams(&resized, gocv.MatTypeCV32F, 1/127.5, -1)
			if ff, err := resized.DataPtrFloat32(); err == nil {
				copy(input.Float32s(), ff)
			}
		} else {
			if v, err := resized.DataPtrUint8(); err == nil {
				copy(input.UInt8s(), v)
			}
		}
		resized.Close()
		frame.Close()

		if interpreter.Invoke() != tflite.OK {
			log.Println("invoke failed")
			return
		}

		latest.Store(&ssdResult{
			loc:   copySlice(interpreter.GetOutputTensor(0).Float32s()),
			clazz: copySlice(interpreter.GetOutputTensor(1).Float32s()),
			score: copySlice(interpreter.GetOutputTensor(2).Float32s()),
		})

		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

func main() {
	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	cam, err := gocv.OpenVideoCapture(*video)
	if err != nil {
		log.Printf("cannot open camera: %v", err)
		return
	}
	defer cam.Close()

	model := tflite.NewModelFromFile(*modelPath)
	if model == nil {
		log.Println("cannot load model")
		return
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	//options.AddDelegate(cl.New(nil))
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

	var wg sync.WaitGroup
	wg.Add(2)

	// Start up the background capture and detection
	frameChan := make(chan gocv.Mat, 1)
	detectChan := make(chan gocv.Mat, 1)
	var latest atomic.Pointer[ssdResult]
	go capture(ctx, &wg, frameChan, detectChan, cam)
	go detect(ctx, &wg, detectChan, &latest, interpreter, wanted_width, wanted_height)

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

	// Buffer terminal output so each frame goes out in as few writes as
	// possible.
	w := bufio.NewWriterSize(os.Stdout, 1024*1024)
	enc := sixel.NewEncoder(w)
	enc.Palette = cubePalette
	enc.Dither = true

	// Clear the screen and hide the cursor while drawing.
	fmt.Print("\x1b[2J\x1b[?25l")
	defer fmt.Print("\x1b[?25h")

	for {
		frame, ok := <-frameChan
		if !ok {
			break
		}

		// Draw the most recent detections onto the frame; they may lag a
		// frame or two behind when inference is slower than the frame rate.
		var classes []ssdClass
		if result := latest.Load(); result != nil {
			classes = make([]ssdClass, 0, len(result.clazz))
			for i := 0; i < len(result.clazz); i++ {
				idx := int(result.clazz[i] + 1)
				score := float64(result.score[i])
				if score < 0.6 {
					continue
				}
				classes = append(classes, ssdClass{loc: result.loc[i*4 : (i+1)*4], score: score, index: idx})
			}
			sort.Slice(classes, func(i, j int) bool {
				return classes[i].score > classes[j].score
			})
			if len(classes) > *limits {
				classes = classes[:*limits]
			}
		}

		size := frame.Size()
		for i, class := range classes {
			label := "unknown"
			if class.index < len(labels) {
				label = labels[class.index]
			}
			c := colornames.Map[colornames.Names[class.index%len(colornames.Names)]]
			gocv.Rectangle(&frame, image.Rect(
				int(float32(size[1])*class.loc[1]),
				int(float32(size[0])*class.loc[0]),
				int(float32(size[1])*class.loc[3]),
				int(float32(size[0])*class.loc[2]),
			), c, 2)
			text := fmt.Sprintf("%d %.5f %s", i, class.score, label)
			gocv.PutText(&frame, text, image.Pt(
				int(float32(size[1])*class.loc[1]),
				int(float32(size[0])*class.loc[0]),
			), gocv.FontHersheySimplex, 1.2, c, 1)
		}

		img, err := frame.ToImage()
		frame.Close()
		if err != nil {
			log.Println(err)
			break
		}

		// Move the cursor to the top-left and draw the frame as sixel.
		fmt.Fprint(w, "\x1b[H")
		enc.Encode(img)
		w.Flush()

		// calculate frame rate
		frames++
		select {
		case <-second:
			// Update the terminal title with the current FPS.
			fmt.Printf("\x1b]0;SSD | FPS: %d\x07", frames)
			frames = 0
		default:
		}
	}

	cancel()
	wg.Wait()
}
