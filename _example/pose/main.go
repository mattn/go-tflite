package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"sync"
	"time"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
)

type vector2d struct {
	x float64
	y float64
}

func (p *vector2d) scale(x, y float64) {
	p.x *= x
	p.y *= y
}

type pose struct {
	keypoints []*keypoint
	score     float64
}

func (p *pose) scale(x, y float64) {
	for i := 0; i < len(p.keypoints); i++ {
		p.keypoints[i].position.scale(x, y)
	}
}

type keypoint struct {
	part     string
	score    float64
	position vector2d
}

type part struct {
	x  int
	y  int
	id int
}

type partWithScore struct {
	score float64
	part  part
}

// tensorData is a view of a NHWC float32 tensor. channelOffset/channelStride
// allow addressing a channel range inside a wider tensor, which is needed for
// models that pack displacement_fwd and displacement_bwd into a single
// mid_offsets tensor.
type tensorData struct {
	data          []float32
	height        int
	width         int
	depth         int
	channelOffset int
	channelStride int
}

func newTensorData(t *tflite.Tensor) *tensorData {
	return &tensorData{
		data:          t.Float32s(),
		height:        t.Dim(1),
		width:         t.Dim(2),
		depth:         t.Dim(3),
		channelStride: t.Dim(3),
	}
}

func (td *tensorData) slice(offset, depth int) *tensorData {
	sliced := *td
	sliced.channelOffset += offset
	sliced.depth = depth
	return &sliced
}

func (td *tensorData) at(y, x, c int) float64 {
	return float64(td.data[(y*td.width+x)*td.channelStride+td.channelOffset+c])
}

func squaredDistance(x1, y1, x2, y2 float64) float64 {
	dy := y2 - y1
	dx := x2 - x1
	return dy*dy + dx*dx
}

func addVectors(a, b vector2d) vector2d {
	return vector2d{x: a.x + b.x, y: a.y + b.y}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func scoreIsMaximumInLocalWindow(id int, score float64, y int, x int, r int, scores *tensorData) bool {
	minmax := true
	ys := max(y-r, 0)
	ye := min(y+r+1, scores.height)

loop:
	for yc := ys; yc < ye; yc++ {
		xs := max(x-r, 0)
		xe := min(x+r+1, scores.width)
		for xc := xs; xc < xe; xc++ {
			if scores.at(yc, xc, id) > score {
				minmax = false
				break loop
			}
		}
	}
	return minmax
}

/**
 * Builds a priority queue with part candidate positions for a specific image in
 * the batch. For this we find all local maxima in the score maps with score
 * values above a threshold. We create a single priority queue across all parts.
 */
func buildPartWithScoreQueue(scoreThreshold float64, r int, scores *tensorData) *MaxHeap {
	queue := NewMaxHeap(scores.height * scores.width * scores.depth)

	for y := 0; y < scores.height; y++ {
		for x := 0; x < scores.width; x++ {
			for i := 0; i < scores.depth; i++ {
				score := scores.at(y, x, i)

				// Only consider parts with score greater or equal to threshold as
				// root candidates.
				if score < scoreThreshold {
					continue
				}

				// Only consider keypoints whose score is maximum in a local window.
				if scoreIsMaximumInLocalWindow(i, score, y, x, r, scores) {
					queue.enqueue(&partWithScore{score: score, part: part{x: x, y: y, id: i}})
				}
			}
		}
	}
	return queue
}

var colors = [17]color.RGBA{
	color.RGBA{R: 255, G: 0, B: 0, A: 255},
	color.RGBA{R: 255, G: 128, B: 0, A: 255},
	color.RGBA{R: 255, G: 0, B: 128, A: 255},
	color.RGBA{R: 255, G: 192, B: 0, A: 255},
	color.RGBA{R: 255, G: 0, B: 192, A: 255},
	color.RGBA{R: 0, G: 255, B: 0, A: 255},
	color.RGBA{R: 0, G: 255, B: 128, A: 255},
	color.RGBA{R: 128, G: 255, B: 0, A: 255},
	color.RGBA{R: 0, G: 255, B: 192, A: 255},
	color.RGBA{R: 192, G: 255, B: 0, A: 255},
	color.RGBA{R: 0, G: 255, B: 255, A: 255},
	color.RGBA{R: 0, G: 0, B: 255, A: 255},
	color.RGBA{R: 128, G: 0, B: 255, A: 255},
	color.RGBA{R: 0, G: 128, B: 255, A: 255},
	color.RGBA{R: 192, G: 0, B: 255, A: 255},
	color.RGBA{R: 0, G: 192, B: 255, A: 255},
	color.RGBA{R: 255, G: 255, B: 0, A: 255},
}

var boneColor = color.RGBA{R: 0, G: 192, B: 255, A: 255}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(x)*(-1))))
}

const minPoseScore = 0.15
const minPartScore = 0.5

func main() {
	var model_path, image_path, video_path string
	flag.StringVar(&model_path, "model", "multi_person_mobilenet_v1_075_float.tflite", "path to model file")
	flag.StringVar(&image_path, "image", "", "path to image file; when set, writes output.png instead of using the camera")
	flag.StringVar(&video_path, "camera", "0", "video capture source (device number or video file)")
	flag.Parse()

	model := tflite.NewModelFromFile(model_path)
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

	if image_path != "" {
		runImage(interpreter, image_path)
	} else {
		runVideo(interpreter, video_path)
	}
}

func runImage(interpreter *tflite.Interpreter, image_path string) {
	f, err := os.Open(image_path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	poses, err := estimateMultiplePoses(
		interpreter,
		img,
		5,
		0.5,
		20)
	if err != nil {
		log.Fatal(err)
	}

	canvas := image.NewRGBA(img.Bounds())
	gc := draw2dimg.NewGraphicContext(canvas)
	gc.DrawImage(img)

	for _, pose := range poses {
		if pose.score < minPoseScore {
			continue
		}
		fmt.Printf("pose score=%f\n", pose.score)
		for _, keypoint := range pose.keypoints {
			fmt.Printf("  %-13s score=%f x=%.1f y=%.1f\n",
				keypoint.part, keypoint.score, keypoint.position.x, keypoint.position.y)
		}

		gc.SetLineWidth(3)
		for _, pair := range parentChildrenTuples {
			p1 := pose.keypoints[pair[0]]
			p2 := pose.keypoints[pair[1]]
			if p1.score < minPartScore || p2.score < minPartScore {
				continue
			}
			gc.SetStrokeColor(boneColor)
			gc.MoveTo(p1.position.x, p1.position.y)
			gc.LineTo(p2.position.x, p2.position.y)
			gc.Stroke()
		}

		for i, keypoint := range pose.keypoints {
			if keypoint.score < minPartScore {
				continue
			}
			gc.SetFillColor(colors[i])
			draw2dkit.Circle(gc, keypoint.position.x, keypoint.position.y, 4)
			gc.Fill()
		}
	}

	err = draw2dimg.SaveToPngFile("output.png", canvas)
	if err != nil {
		log.Println(err)
	}
}

type poseResult struct {
	poses []pose
	mat   gocv.Mat
}

func detect(ctx context.Context, wg *sync.WaitGroup, resultChan chan<- *poseResult, interpreter *tflite.Interpreter, cam *gocv.VideoCapture) {
	defer wg.Done()
	defer close(resultChan)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		frame := gocv.NewMat()
		if ok := cam.Read(&frame); !ok {
			frame.Close()
			break
		}

		img, err := frame.ToImage()
		if err != nil {
			frame.Close()
			continue
		}

		poses, err := estimateMultiplePoses(interpreter, img, 5, 0.5, 20)
		if err != nil {
			frame.Close()
			log.Println(err)
			return
		}

		resultChan <- &poseResult{poses: poses, mat: frame}
	}
}

func drawPoses(mat *gocv.Mat, poses []pose) {
	for _, pose := range poses {
		if pose.score < minPoseScore {
			continue
		}
		for _, pair := range parentChildrenTuples {
			p1 := pose.keypoints[pair[0]]
			p2 := pose.keypoints[pair[1]]
			if p1.score < minPartScore || p2.score < minPartScore {
				continue
			}
			gocv.Line(mat,
				image.Pt(int(p1.position.x), int(p1.position.y)),
				image.Pt(int(p2.position.x), int(p2.position.y)),
				boneColor, 2)
		}
		for i, keypoint := range pose.keypoints {
			if keypoint.score < minPartScore {
				continue
			}
			gocv.Circle(mat,
				image.Pt(int(keypoint.position.x), int(keypoint.position.y)),
				4, colors[i], -1)
		}
	}
}

func runVideo(interpreter *tflite.Interpreter, video_path string) {
	cam, err := gocv.OpenVideoCapture(video_path)
	if err != nil {
		log.Printf("cannot open camera: %v", err)
		return
	}
	defer cam.Close()

	window := gocv.NewWindow("Pose")
	defer window.Close()
	window.ResizeWindow(
		int(cam.Get(gocv.VideoCaptureFrameWidth)),
		int(cam.Get(gocv.VideoCaptureFrameHeight)),
	)

	ctx, cancel := context.WithCancel(context.Background())

	var wg sync.WaitGroup
	wg.Add(1)
	resultChan := make(chan *poseResult, 1)
	go detect(ctx, &wg, resultChan, interpreter, cam)

	sc := make(chan os.Signal, 1)
	defer close(sc)
	signal.Notify(sc, os.Interrupt)
	go func() {
		<-sc
		cancel()
	}()

	frames := 0
	second := time.Tick(time.Second)

	for {
		result, ok := <-resultChan
		if !ok {
			break
		}

		drawPoses(&result.mat, result.poses)

		window.IMShow(result.mat)
		result.mat.Close()

		k := window.WaitKey(1)
		if k == 0x1b {
			break
		}
		if window.GetWindowProperty(gocv.WindowPropertyVisible) == 0 {
			break
		}

		frames++
		select {
		case <-second:
			window.SetWindowTitle(fmt.Sprintf("Pose | FPS: %d", frames))
			frames = 0
		default:
		}
	}

	cancel()
	for {
		if result, ok := <-resultChan; ok {
			result.mat.Close()
		} else {
			break
		}
	}
	wg.Wait()
}

func getOffsetPoint(y int, x int, i int, offsets *tensorData) vector2d {
	numParts := offsets.depth / 2
	return vector2d{
		y: offsets.at(y, x, i),
		x: offsets.at(y, x, i+numParts),
	}
}

func getImageCoords(p part, outputStride int, offsets *tensorData) vector2d {
	pos := getOffsetPoint(p.y, p.x, p.id, offsets)
	return vector2d{
		x: float64(p.x*outputStride) + pos.x,
		y: float64(p.y*outputStride) + pos.y,
	}
}

var partNames = []string{
	"nose",
	"leftEye",
	"rightEye",
	"leftEar",
	"rightEar",
	"leftShoulder",
	"rightShoulder",
	"leftElbow",
	"rightElbow",
	"leftWrist",
	"rightWrist",
	"leftHip",
	"rightHip",
	"leftKnee",
	"rightKnee",
	"leftAnkle",
	"rightAnkle",
}

var parentChildrenTuples = [][2]int{
	{0, 1},
	{1, 3},
	{0, 2},
	{2, 4},
	{0, 5},
	{5, 7},
	{7, 9},
	{5, 11},
	{11, 13},
	{13, 15},
	{0, 6},
	{6, 8},
	{8, 10},
	{6, 12},
	{12, 14},
	{14, 16},
}

var parentToChildEdges = []int{1, 3, 2, 4, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16}

var childToParentEdges = []int{0, 1, 0, 2, 0, 5, 7, 5, 11, 13, 0, 6, 8, 6, 12, 14}

func clamp(a, min, max float64) float64 {
	if a < min {
		return min
	}
	if a > max {
		return max
	}
	return a
}

func getStridedIndexNearPoint(point vector2d, outputStride int, height int, width int) vector2d {
	return vector2d{
		y: clamp(math.Round(point.y/float64(outputStride)), 0, float64(height-1)),
		x: clamp(math.Round(point.x/float64(outputStride)), 0, float64(width-1)),
	}
}

func getDisplacement(edgeId int, point vector2d, displacements *tensorData) vector2d {
	numEdges := displacements.depth / 2
	return vector2d{
		y: displacements.at(int(point.y), int(point.x), edgeId),
		x: displacements.at(int(point.y), int(point.x), edgeId+numEdges),
	}
}

const offsetRefineStep = 2

func traverseToTargetKeypoint(edgeId int, sourceKeypoint *keypoint, targetKeypointId int, scores, offsets *tensorData, outputStride int, displacements *tensorData) *keypoint {
	height := scores.height
	width := scores.width
	// Nearest neighbor interpolation for the source->target displacements.
	sourceKeypointIndices := getStridedIndexNearPoint(sourceKeypoint.position, outputStride, height, width)
	displacement := getDisplacement(edgeId, sourceKeypointIndices, displacements)
	targetKeypoint := addVectors(sourceKeypoint.position, displacement)
	for i := 0; i < offsetRefineStep; i++ {
		targetKeypointIndices := getStridedIndexNearPoint(targetKeypoint, outputStride, height, width)
		offsetPoint := getOffsetPoint(int(targetKeypointIndices.y), int(targetKeypointIndices.x), targetKeypointId, offsets)
		targetKeypoint = addVectors(vector2d{
			x: targetKeypointIndices.x * float64(outputStride),
			y: targetKeypointIndices.y * float64(outputStride),
		}, offsetPoint)
	}
	targetKeypointIndices := getStridedIndexNearPoint(targetKeypoint, outputStride, height, width)
	score := scores.at(int(targetKeypointIndices.y), int(targetKeypointIndices.x), targetKeypointId)
	return &keypoint{
		position: targetKeypoint,
		part:     partNames[targetKeypointId],
		score:    score,
	}
}

func decodePose(
	root *partWithScore,
	scores *tensorData,
	offsets *tensorData,
	outputStride int,
	displacementsFwd *tensorData,
	displacementsBwd *tensorData) []*keypoint {

	numParts := scores.depth
	numEdges := len(parentToChildEdges)

	instanceKeypoints := make([]*keypoint, numParts)
	// Start a new detection instance at the position of the root.
	rootPart := root.part
	rootScore := root.score
	rootPoint := getImageCoords(rootPart, outputStride, offsets)

	instanceKeypoints[rootPart.id] = &keypoint{
		score:    rootScore,
		part:     partNames[rootPart.id],
		position: rootPoint,
	}

	// Decode the part positions upwards in the tree, following the backward
	// displacements.
	for edge := numEdges - 1; edge >= 0; edge-- {
		sourceKeypointId := parentToChildEdges[edge]
		targetKeypointId := childToParentEdges[edge]
		if instanceKeypoints[sourceKeypointId] != nil &&
			instanceKeypoints[targetKeypointId] == nil {
			instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
				edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
				offsets, outputStride, displacementsBwd)
		}
	}

	// Decode the part positions downwards in the tree, following the forward
	// displacements.
	for edge := 0; edge < numEdges; edge++ {
		sourceKeypointId := childToParentEdges[edge]
		targetKeypointId := parentToChildEdges[edge]
		if instanceKeypoints[sourceKeypointId] != nil && instanceKeypoints[targetKeypointId] == nil {
			instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsFwd)
		}
	}

	return instanceKeypoints
}

const kLocalMaximumRadius = 1

func withinNmsRadiusOfCorrespondingPoint(
	poses []pose, squaredNmsRadius int, pos vector2d, keypointId int) bool {
	for i := 0; i < len(poses); i++ {
		correspondingKeypoint := poses[i].keypoints[keypointId].position
		if squaredDistance(float64(pos.y), float64(pos.x), float64(correspondingKeypoint.y), float64(correspondingKeypoint.x)) <= float64(squaredNmsRadius) {
			return true
		}
	}
	return false
}

func getInstanceScore(existingPoses []pose, squaredNmsRadius int, instanceKeypoints []*keypoint) float64 {
	notOverlappedKeypointScores := float64(0)
	for i := 0; i < len(instanceKeypoints); i++ {
		keypoint := instanceKeypoints[i]
		if keypoint == nil {
			continue
		}
		if !withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, keypoint.position, i) {
			notOverlappedKeypointScores += keypoint.score
		}
	}
	return notOverlappedKeypointScores / float64(len(instanceKeypoints))
}

func half(k int) int {
	return k / 2
}

type MaxHeap struct {
	priorityQueue    []*partWithScore
	numberOfElements int
}

func NewMaxHeap(maxSize int) *MaxHeap {
	return &MaxHeap{
		priorityQueue:    make([]*partWithScore, maxSize),
		numberOfElements: -1,
	}
}

func (h *MaxHeap) enqueue(x *partWithScore) {
	h.numberOfElements++
	h.priorityQueue[h.numberOfElements] = x
	h.swim(h.numberOfElements)
}

func (h *MaxHeap) dequeue() *partWithScore {
	max := h.priorityQueue[0]
	h.exchange(0, h.numberOfElements)
	h.numberOfElements--
	h.sink(0)
	h.priorityQueue[h.numberOfElements+1] = nil
	return max
}

func (h *MaxHeap) empty() bool {
	return h.numberOfElements == -1
}

func (h *MaxHeap) size() int {
	return h.numberOfElements + 1
}

func (h *MaxHeap) all() []*partWithScore {
	return h.priorityQueue[0 : h.numberOfElements+1]
}

func (h *MaxHeap) max() *partWithScore {
	return h.priorityQueue[0]
}

func (h *MaxHeap) swim(k int) {
	for k > 0 && h.less(half(k), k) {
		h.exchange(k, half(k))
		k = half(k)
	}
}

func (h *MaxHeap) sink(k int) {
	for 2*k <= h.numberOfElements {
		var j = 2 * k
		if j < h.numberOfElements && h.less(j, j+1) {
			j++
		}
		if !h.less(k, j) {
			break
		}
		h.exchange(k, j)
		k = j
	}
}

func (h *MaxHeap) getValueAt(i int) float64 {
	return h.priorityQueue[i].score
}

func (h *MaxHeap) less(i, j int) bool {
	return h.getValueAt(i) < h.getValueAt(j)
}

func (h *MaxHeap) exchange(i, j int) {
	h.priorityQueue[i], h.priorityQueue[j] = h.priorityQueue[j], h.priorityQueue[i]
}

func decodeMultiplePoses(
	scores *tensorData,
	offsets *tensorData,
	displacementsFwd *tensorData,
	displacementsBwd *tensorData,
	outputStride int,
	maxPoseDetections int,
	scoreThreshold float64,
	nmsRadius int) []pose {

	poses := []pose{}

	queue := buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, scores)

	squaredNmsRadius := nmsRadius * nmsRadius

	// Generate at most maxDetections object instances per image in
	// decreasing root part score order.
	for len(poses) < maxPoseDetections && !queue.empty() {
		// The top element in the queue is the next root candidate.
		root := queue.dequeue()

		// Part-based non-maximum suppression: We reject a root candidate if it
		// is within a disk of `nmsRadius` pixels from the corresponding part of
		// a previously detected instance.
		rootImageCoords := getImageCoords(root.part, outputStride, offsets)
		if withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root.part.id) {
			continue
		}

		// Start a new detection instance at the position of the root.
		keypoints := decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd)

		score := getInstanceScore(poses, squaredNmsRadius, keypoints)

		poses = append(poses, pose{keypoints: keypoints, score: score})
	}

	return poses
}

func sigmoidTensorData(td *tensorData) {
	for i, f := range td.data {
		td.data[i] = sigmoid(f)
	}
}

// findOutputTensors locates the heatmap, offset and displacement tensors by
// their channel counts, so both flavors of the PoseNet model work: models with
// separate displacement_fwd/displacement_bwd tensors (2*16 channels each), and
// models with a single mid_offsets tensor packing both (4*16 channels).
func findOutputTensors(interpreter *tflite.Interpreter) (scores, offsets, displacementsFwd, displacementsBwd *tensorData, err error) {
	numParts := len(partNames)
	numEdges := len(parentToChildEdges)
	for i := 0; i < interpreter.GetOutputTensorCount(); i++ {
		t := interpreter.GetOutputTensor(i)
		if t.NumDims() != 4 || t.Type() != tflite.Float32 {
			continue
		}
		switch t.Dim(3) {
		case numParts:
			scores = newTensorData(t)
		case 2 * numParts:
			offsets = newTensorData(t)
		case 2 * numEdges:
			if displacementsFwd == nil {
				displacementsFwd = newTensorData(t)
			} else {
				displacementsBwd = newTensorData(t)
			}
		case 4 * numEdges:
			midOffsets := newTensorData(t)
			displacementsFwd = midOffsets.slice(0, 2*numEdges)
			displacementsBwd = midOffsets.slice(2*numEdges, 2*numEdges)
		}
	}
	if scores == nil || offsets == nil || displacementsFwd == nil || displacementsBwd == nil {
		return nil, nil, nil, nil, errors.New("unexpected model outputs: not a PoseNet model?")
	}
	return scores, offsets, displacementsFwd, displacementsBwd, nil
}

func estimateMultiplePoses(
	interpreter *tflite.Interpreter,
	img image.Image, maxDetections int, scoreThreshold float64,
	nmsRadius int) ([]pose, error) {

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)

	resized := resize.Resize(uint(wanted_width), uint(wanted_height), img, resize.Bilinear)
	ff := input.Float32s()
	for y := 0; y < wanted_height; y++ {
		for x := 0; x < wanted_width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			ff[(y*wanted_width+x)*3+0] = (float32(r)/256 - 127.5) / 127.5
			ff[(y*wanted_width+x)*3+1] = (float32(g)/256 - 127.5) / 127.5
			ff[(y*wanted_width+x)*3+2] = (float32(b)/256 - 127.5) / 127.5
		}
	}

	status := interpreter.Invoke()
	if status != tflite.OK {
		return nil, errors.New("invoke failed")
	}

	scores, offsets, displacementsFwd, displacementsBwd, err := findOutputTensors(interpreter)
	if err != nil {
		return nil, err
	}
	sigmoidTensorData(scores)

	outputStride := (wanted_height - 1) / (scores.height - 1)

	poses := decodeMultiplePoses(
		scores, offsets, displacementsFwd, displacementsBwd, outputStride,
		maxDetections, scoreThreshold, nmsRadius)

	scaleY := float64(img.Bounds().Dy()) / float64(wanted_height)
	scaleX := float64(img.Bounds().Dx()) / float64(wanted_width)
	for i := 0; i < len(poses); i++ {
		poses[i].scale(scaleX, scaleY)
	}
	return poses, nil
}
