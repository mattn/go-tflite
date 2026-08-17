package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"sort"

	"github.com/mattn/go-tflite"
	"gocv.io/x/gocv"
)

const (
	detectionSize      = 320
	scoreThreshold     = 0.5
	nmsThreshold       = 0.4
	recognitionWidth   = 200
	recognitionHeight  = 31
	recognitionEpsilon = 1e-6
)

// keras-ocr alphabet; the last class (index 36) is the CTC blank.
const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

var detectionMeans = [3]float32{103.94, 116.78, 123.68}

type detection struct {
	corners [4]image.Point // bl, tl, tr, br in original image coords
	rect    image.Rectangle
	score   float32
}

func newInterpreter(model_path string) (*tflite.Model, *tflite.Interpreter, error) {
	model := tflite.NewModelFromFile(model_path)
	if model == nil {
		return nil, nil, fmt.Errorf("cannot load model %s", model_path)
	}
	options := tflite.NewInterpreterOptions()
	options.SetNumThread(4)
	defer options.Delete()
	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		model.Delete()
		return nil, nil, fmt.Errorf("cannot create interpreter for %s", model_path)
	}
	if interpreter.AllocateTensors() != tflite.OK {
		interpreter.Delete()
		model.Delete()
		return nil, nil, fmt.Errorf("allocate failed for %s", model_path)
	}
	return model, interpreter, nil
}

// detectTexts runs the EAST detector and returns text boxes in original
// image coordinates, following the decoding in the official TensorFlow Lite
// OCR example.
func detectTexts(interpreter *tflite.Interpreter, img gocv.Mat) []detection {
	size := img.Size()
	ratioX := float64(size[1]) / detectionSize
	ratioY := float64(size[0]) / detectionSize

	resized := gocv.NewMat()
	defer resized.Close()
	gocv.CvtColor(img, &resized, gocv.ColorBGRToRGB)
	gocv.Resize(resized, &resized, image.Pt(detectionSize, detectionSize), 0, 0, gocv.InterpolationDefault)
	resized.ConvertTo(&resized, gocv.MatTypeCV32F)

	in := interpreter.GetInputTensor(0).Float32s()
	ff, err := resized.DataPtrFloat32()
	if err != nil {
		log.Fatal(err)
	}
	for i, v := range ff {
		in[i] = v - detectionMeans[i%3]
	}

	if interpreter.Invoke() != tflite.OK {
		log.Fatal("invoke failed")
	}

	// Locate the score [1,H,W,1] and geometry [1,H,W,5] tensors.
	var scores, geometry []float32
	var gridH, gridW int
	for i := 0; i < interpreter.GetOutputTensorCount(); i++ {
		t := interpreter.GetOutputTensor(i)
		if t.NumDims() != 4 {
			continue
		}
		switch t.Dim(3) {
		case 1:
			scores = t.Float32s()
			gridH, gridW = t.Dim(1), t.Dim(2)
		case 5:
			geometry = t.Float32s()
		}
	}
	if scores == nil || geometry == nil {
		log.Fatal("unexpected detection model outputs")
	}

	var candidates []detection
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			score := scores[y*gridW+x]
			if score < scoreThreshold {
				continue
			}
			g := geometry[(y*gridW+x)*5:]
			offsetX, offsetY := float64(x)*4, float64(y)*4
			a := float64(g[4])
			h := float64(g[0] + g[2])
			w := float64(g[1] + g[3])
			ox := offsetX + math.Cos(a)*float64(g[1]) + math.Sin(a)*float64(g[2])
			oy := offsetY - math.Sin(a)*float64(g[1]) + math.Cos(a)*float64(g[2])
			p1 := [2]float64{-math.Sin(a)*h + ox, -math.Cos(a)*h + oy}
			p3 := [2]float64{-math.Cos(a)*w + ox, math.Sin(a)*w + oy}
			cx, cy := (p1[0]+p3[0])/2, (p1[1]+p3[1])/2

			// corners of the rotated rect (OpenCV boxPoints order:
			// bottom-left, top-left, top-right, bottom-right)
			ca, sa := math.Cos(-a)*0.5, math.Sin(-a)*0.5
			var corners [4]image.Point
			corners[0] = image.Pt(
				int((cx-sa*h-ca*w)*ratioX), int((cy+ca*h-sa*w)*ratioY))
			corners[1] = image.Pt(
				int((cx+sa*h-ca*w)*ratioX), int((cy-ca*h-sa*w)*ratioY))
			corners[2] = image.Pt(
				int(2*cx*ratioX)-corners[0].X, int(2*cy*ratioY)-corners[0].Y)
			corners[3] = image.Pt(
				int(2*cx*ratioX)-corners[1].X, int(2*cy*ratioY)-corners[1].Y)

			rect := image.Rect(corners[0].X, corners[0].Y, corners[0].X+1, corners[0].Y+1)
			for _, c := range corners[1:] {
				rect = rect.Union(image.Rect(c.X, c.Y, c.X+1, c.Y+1))
			}
			candidates = append(candidates, detection{corners: corners, rect: rect, score: score})
		}
	}

	// greedy NMS on the bounding rectangles
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	var result []detection
	for _, c := range candidates {
		keep := true
		for _, r := range result {
			inter := c.rect.Intersect(r.rect)
			ia := inter.Dx() * inter.Dy()
			ua := c.rect.Dx()*c.rect.Dy() + r.rect.Dx()*r.rect.Dy() - ia
			if ua > 0 && float64(ia)/float64(ua) > nmsThreshold {
				keep = false
				break
			}
		}
		if keep {
			result = append(result, c)
		}
	}
	return result
}

// recognizeText warps one detected box to the recognition input size and
// greedy-decodes the CTC output.
func recognizeText(interpreter *tflite.Interpreter, img gocv.Mat, d detection) string {
	src := gocv.NewPointVectorFromPoints([]image.Point{
		d.corners[1], d.corners[2], d.corners[3], d.corners[0], // tl, tr, br, bl
	})
	defer src.Close()
	dst := gocv.NewPointVectorFromPoints([]image.Point{
		image.Pt(0, 0),
		image.Pt(recognitionWidth-1, 0),
		image.Pt(recognitionWidth-1, recognitionHeight-1),
		image.Pt(0, recognitionHeight-1),
	})
	defer dst.Close()

	m := gocv.GetPerspectiveTransform(src, dst)
	defer m.Close()
	warped := gocv.NewMat()
	defer warped.Close()
	gocv.WarpPerspective(img, &warped, m, image.Pt(recognitionWidth, recognitionHeight))

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(warped, &gray, gocv.ColorBGRToGray)
	gray.ConvertToWithParams(&gray, gocv.MatTypeCV32F, 1.0/255, 0)

	ff, err := gray.DataPtrFloat32()
	if err != nil {
		log.Fatal(err)
	}
	copy(interpreter.GetInputTensor(0).Float32s(), ff)

	if interpreter.Invoke() != tflite.OK {
		log.Fatal("invoke failed")
	}

	output := interpreter.GetOutputTensor(0)
	steps, classes := output.Dim(1), output.Dim(2)
	v := output.Float32s()
	blank := classes - 1
	text := ""
	prev := -1
	for t := 0; t < steps; t++ {
		best, bestV := 0, v[t*classes]
		for c := 1; c < classes; c++ {
			if v[t*classes+c] > bestV {
				best, bestV = c, v[t*classes+c]
			}
		}
		if best != prev && best != blank && best < len(alphabet) {
			text += string(alphabet[best])
		}
		prev = best
	}
	return text
}

func main() {
	var detection_model, recognition_model, image_path, out_path string
	flag.StringVar(&detection_model, "detection", "east_text_detector.tflite", "path to text detection model")
	flag.StringVar(&recognition_model, "recognition", "text_recognition.tflite", "path to text recognition model")
	flag.StringVar(&image_path, "image", "test.jpg", "path to image file")
	flag.StringVar(&out_path, "out", "output.png", "path to output image")
	flag.Parse()

	img := gocv.IMRead(image_path, gocv.IMReadColor)
	if img.Empty() {
		log.Fatalf("cannot read %s", image_path)
	}
	defer img.Close()

	dModel, dInterp, err := newInterpreter(detection_model)
	if err != nil {
		log.Fatal(err)
	}
	defer dModel.Delete()
	defer dInterp.Delete()

	rModel, rInterp, err := newInterpreter(recognition_model)
	if err != nil {
		log.Fatal(err)
	}
	defer rModel.Delete()
	defer rInterp.Delete()

	green := color.RGBA{G: 255, A: 255}
	for _, d := range detectTexts(dInterp, img) {
		text := recognizeText(rInterp, img, d)
		if text == "" {
			continue
		}
		fmt.Printf("%.2f %-20s %v\n", d.score, text, d.rect)
		for i := range d.corners {
			gocv.Line(&img, d.corners[i], d.corners[(i+1)%4], green, 2)
		}
		gocv.PutText(&img, text, image.Pt(d.rect.Min.X, d.rect.Min.Y-4),
			gocv.FontHersheySimplex, 0.8, green, 2)
	}

	if ok := gocv.IMWrite(out_path, img); !ok {
		log.Fatalf("cannot write %s", out_path)
	}
}
