package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"sort"

	"github.com/mattn/go-tflite"
	"gocv.io/x/gocv"
)

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

// MoViNet stream models carry their recurrent state in dozens of state
// tensors that must be fed back into the next invocation. Most state outputs
// share their tensor with the same-named input, so feeding back is a no-op
// for them, but for these the output is a distinct tensor that has to be
// copied to the input. The mapping comes from the model's SignatureDef,
// which go-tflite cannot read, so it is spelled out here (MoViNet-A0 stream).
var stateFeedback = map[string]string{
	"StatefulPartitionedCall:5":  "serving_default_state_block1_layer0_stream_buffer:0",
	"StatefulPartitionedCall:8":  "serving_default_state_block1_layer1_stream_buffer:0",
	"StatefulPartitionedCall:11": "serving_default_state_block1_layer2_stream_buffer:0",
	"StatefulPartitionedCall:14": "serving_default_state_block2_layer0_stream_buffer:0",
	"StatefulPartitionedCall:17": "serving_default_state_block2_layer1_stream_buffer:0",
	"StatefulPartitionedCall:20": "serving_default_state_block2_layer2_stream_buffer:0",
	"StatefulPartitionedCall:23": "serving_default_state_block3_layer0_stream_buffer:0",
	"StatefulPartitionedCall:26": "serving_default_state_block3_layer1_stream_buffer:0",
	"StatefulPartitionedCall:29": "serving_default_state_block3_layer2_stream_buffer:0",
	"StatefulPartitionedCall:32": "serving_default_state_block3_layer3_stream_buffer:0",
	"StatefulPartitionedCall:35": "serving_default_state_block4_layer0_stream_buffer:0",
	"StatefulPartitionedCall:42": "serving_default_state_head_pool_buffer:0",
	"StatefulPartitionedCall:43": "serving_default_state_head_pool_frame_count:0",
}

const modelFPS = 5 // the official example feeds the model 5 frames per second

func softmax(logits []float32) []float64 {
	m := float64(logits[0])
	for _, v := range logits {
		if float64(v) > m {
			m = float64(v)
		}
	}
	sum := 0.0
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(float64(v) - m)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func main() {
	var model_path, label_path, video_path string
	flag.StringVar(&model_path, "model", "movinet_a0_stream.tflite", "path to model file")
	flag.StringVar(&label_path, "label", "kinetics600_label_map.txt", "path to label file")
	flag.StringVar(&video_path, "video", "", "path to video file")
	flag.Parse()

	if video_path == "" {
		flag.Usage()
		os.Exit(1)
	}

	labels, err := loadLabels(label_path)
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

	if interpreter.AllocateTensors() != tflite.OK {
		log.Fatal("allocate failed")
	}

	inputByName := map[string]*tflite.Tensor{}
	var imageInput *tflite.Tensor
	for i := 0; i < interpreter.GetInputTensorCount(); i++ {
		t := interpreter.GetInputTensor(i)
		inputByName[t.Name()] = t
		// The image input is the only float32 input with 3 channels; state
		// buffers are wider. Zero-init everything else.
		switch t.Type() {
		case tflite.Float32:
			buf := t.Float32s()
			for j := range buf {
				buf[j] = 0
			}
			if t.NumDims() == 5 && t.Dim(4) == 3 {
				imageInput = t
			}
		case tflite.Int32:
			buf := t.Int32s()
			for j := range buf {
				buf[j] = 0
			}
		}
	}
	if imageInput == nil {
		log.Fatal("cannot find image input tensor: not a MoViNet stream model?")
	}
	wanted_height := imageInput.Dim(2)
	wanted_width := imageInput.Dim(3)

	var logits *tflite.Tensor
	for i := 0; i < interpreter.GetOutputTensorCount(); i++ {
		t := interpreter.GetOutputTensor(i)
		if t.Name() == "StatefulPartitionedCall:0" {
			logits = t
		}
	}
	if logits == nil {
		log.Fatal("cannot find logits output tensor")
	}

	cam, err := gocv.OpenVideoCapture(video_path)
	if err != nil {
		log.Fatal(err)
	}
	defer cam.Close()

	fps := cam.Get(gocv.VideoCaptureFPS)
	if fps <= 0 {
		fps = 30
	}
	step := int(fps/modelFPS + 0.5)
	if step < 1 {
		step = 1
	}

	frame := gocv.NewMat()
	defer frame.Close()
	resized := gocv.NewMat()
	defer resized.Close()

	var probs []float64
	n := 0
	for ; cam.Read(&frame); n++ {
		if n%step != 0 {
			continue
		}

		// center-crop to square, resize, RGB in 0..1
		size := frame.Size()
		side := size[0]
		if size[1] < side {
			side = size[1]
		}
		crop := frame.Region(image.Rect(
			(size[1]-side)/2, (size[0]-side)/2,
			(size[1]-side)/2+side, (size[0]-side)/2+side))
		gocv.CvtColor(crop, &resized, gocv.ColorBGRToRGB)
		crop.Close()
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		resized.ConvertToWithParams(&resized, gocv.MatTypeCV32F, 1.0/255, 0)
		if ff, err := resized.DataPtrFloat32(); err == nil {
			copy(imageInput.Float32s(), ff)
		}

		if interpreter.Invoke() != tflite.OK {
			log.Fatal("invoke failed")
		}

		// feed the recurrent state back
		for i := 0; i < interpreter.GetOutputTensorCount(); i++ {
			out := interpreter.GetOutputTensor(i)
			name := out.Name()
			if _, inPlace := inputByName[name]; inPlace {
				continue // output shares the input tensor
			}
			inName, ok := stateFeedback[name]
			if !ok {
				continue // logits
			}
			in, ok := inputByName[inName]
			if !ok {
				log.Fatalf("state input %q not found", inName)
			}
			switch out.Type() {
			case tflite.Float32:
				copy(in.Float32s(), out.Float32s())
			case tflite.Int32:
				copy(in.Int32s(), out.Int32s())
			}
		}

		probs = softmax(logits.Float32s())
	}

	if probs == nil {
		log.Fatal("no frames read")
	}

	type result struct {
		label string
		prob  float64
	}
	results := make([]result, len(probs))
	for i := range probs {
		label := "unknown"
		if i < len(labels) {
			label = labels[i]
		}
		results[i] = result{label: label, prob: probs[i]}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].prob > results[j].prob
	})
	fmt.Printf("%d frames fed (every %d frames)\n", (n+step-1)/step, step)
	for _, r := range results[:5] {
		fmt.Printf("  %-30s %.3f\n", r.label, r.prob)
	}
}
