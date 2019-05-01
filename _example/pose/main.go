package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
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

func scoreIsMaximumInLocalWindow(id int, score float64, y int, x int, r int, scores *tflite.Tensor) bool {
	minmax := true
	ys := max(y-r, 0)
	ye := min(y+r+1, scores.Dim(1))

loop:
	for yc := ys; yc < ye; yc++ {
		xs := max(x-r, 0)
		xe := min(x+r+1, scores.Dim(2))
		for xc := xs; xc < xe; xc++ {
			if float64(scores.Float32At(0, yc, xc, id)) > score {
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
func buildPartWithScoreQueue(scoreThreshold float64, r int, scores *tflite.Tensor) *MaxHeap {
	queue := NewMaxHeap(scores.Dim(1) * scores.Dim(2) * scores.Dim(3))

	for y := 0; y < scores.Dim(1); y++ {
		for x := 0; x < scores.Dim(2); x++ {
			for i := 0; i < scores.Dim(3); i++ {
				score := float64(scores.Float32At(0, y, x, i))

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

var colors = [21]color.RGBA{
	color.RGBA{R: 0, G: 0, B: 0, A: 100},
	color.RGBA{R: 128, G: 0, B: 0, A: 100},
	color.RGBA{R: 0, G: 128, B: 0, A: 100},
	color.RGBA{R: 128, G: 128, B: 0, A: 100},
	color.RGBA{R: 0, G: 0, B: 128, A: 100},
	color.RGBA{R: 128, G: 0, B: 128, A: 100},
	color.RGBA{R: 0, G: 128, B: 128, A: 100},
	color.RGBA{R: 128, G: 128, B: 128, A: 100},
	color.RGBA{R: 64, G: 0, B: 0, A: 100},
	color.RGBA{R: 192, G: 0, B: 0, A: 100},
	color.RGBA{R: 64, G: 128, B: 0, A: 100},
	color.RGBA{R: 192, G: 128, B: 0, A: 100},
	color.RGBA{R: 64, G: 0, B: 128, A: 100},
	color.RGBA{R: 192, G: 0, B: 128, A: 100},
	color.RGBA{R: 64, G: 128, B: 128, A: 100},
	color.RGBA{R: 192, G: 128, B: 128, A: 100},
	color.RGBA{R: 0, G: 64, B: 0, A: 100},
	color.RGBA{R: 128, G: 64, B: 0, A: 100},
	color.RGBA{R: 0, G: 192, B: 0, A: 100},
	color.RGBA{R: 128, G: 192, B: 0, A: 100},
	color.RGBA{R: 0, G: 64, B: 128, A: 100},
}

type Circle struct {
	p image.Point
	r int
	c color.Color
}

func (c *Circle) ColorModel() color.Model {
	return color.AlphaModel
}

func (c *Circle) Bounds() image.Rectangle {
	return image.Rect(c.p.X-c.r, c.p.Y-c.r, c.p.X+c.r, c.p.Y+c.r)
}

func (c *Circle) At(x, y int) color.Color {
	xx, yy, rr := float64(x-c.p.X)+0.5, float64(y-c.p.Y)+0.5, float64(c.r)
	if xx*xx+yy*yy < rr*rr {
		return c.c
	}
	return color.Alpha{0}
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(x)*(-1))))
}

func main() {
	var model_path, image_path string
	flag.StringVar(&model_path, "model", "multi_person_mobilenet_v1_075_float.tflite", "path to model file")
	flag.StringVar(&image_path, "image", "aa.png", "path to image file")
	flag.Parse()

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

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Fatal("allocate failed")
	}

	f, err := os.Open(image_path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	poses := estimateMultiplePoses(
		interpreter,
		img,
		0.5,
		false,
		16,
		5,
		0.5,
		30)

	println(len(poses))
	for _, pose := range poses {
		for _, keypoint := range pose.keypoints {
			fmt.Println(keypoint.part)
			fmt.Println(keypoint.score, keypoint.position.x, keypoint.position.y)
		}
	}

	canvas := image.NewRGBA(img.Bounds())
	gc := draw2dimg.NewGraphicContext(canvas)
	gc.DrawImage(img)

	for _, pose := range poses {
		pos := func(i int) (float64, float64) {
			p := pose.keypoints[i].position
			return float64(p.x), float64(p.y)
		}
		center := func(i, j int) (float64, float64) {
			p1 := pose.keypoints[i].position
			p2 := pose.keypoints[j].position
			x := (p1.x + p2.x) / 2
			y := (p1.y + p2.y) / 2
			return float64(x), float64(y)
		}

		for i := range pose.keypoints {
			x, y := pos(i)
			gc.SetFillColor(colors[i])
			draw2dkit.RoundedRectangle(gc, x-5, y-5, x+5, y+5, 10, 10)
			gc.Fill()
		}
		gc.SetLineWidth(5)
		gc.SetStrokeColor(colors[2])

		gc.MoveTo(pos(0))
		gc.LineTo(pos(1))
		gc.LineTo(pos(3))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(pos(0))
		gc.LineTo(pos(2))
		gc.LineTo(pos(4))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(pos(0))
		gc.LineTo(center(5, 6))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(center(5, 6))
		gc.LineTo(pos(5))
		gc.LineTo(pos(7))
		gc.LineTo(pos(9))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(center(5, 6))
		gc.LineTo(pos(6))
		gc.LineTo(pos(8))
		gc.LineTo(pos(10))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(center(5, 6))
		gc.LineTo(center(11, 12))
		gc.Stroke()
		gc.Close()

		gc.MoveTo(center(11, 12))
		gc.LineTo(pos(11))
		gc.LineTo(pos(13))
		if len(pose.keypoints) > 15 {
			gc.LineTo(pos(15))
		}
		gc.Stroke()
		gc.Close()

		gc.MoveTo(center(11, 12))
		gc.LineTo(pos(12))
		if len(pose.keypoints) > 14 {
			gc.LineTo(pos(14))
		}
		if len(pose.keypoints) > 16 {
			gc.LineTo(pos(16))
		}
		gc.Stroke()
		gc.Close()
	}

	err = draw2dimg.SaveToPngFile("output.png", canvas)
	if err != nil {
		log.Fatal(err)
	}
}

func getOffsetPoint(y int, x int, i int, offsets *tflite.Tensor) vector2d {
	return vector2d{
		y: float64(offsets.Float32At(0, y, x, i)),
		x: float64(offsets.Float32At(0, y, x, i+17)),
	}
}

func getImageCoords(p part, outputStride int, offsets *tflite.Tensor) vector2d {
	pos := getOffsetPoint(p.y, p.x, p.id, offsets)
	return vector2d{
		x: float64(p.x*outputStride) + pos.x,
		y: float64(p.y*outputStride) + pos.y,
	}
}

var poseChain = [][2]string{
	{"nose", "leftEye"},
	{"leftEye", "leftEar"},
	{"nose", "rightEye"},
	{"rightEye", "rightEar"},
	{"nose", "leftShoulder"},
	{"leftShoulder", "leftElbow"},
	{"leftElbow", "leftWrist"},
	{"leftShoulder", "leftHip"},
	{"leftHip", "leftKnee"},
	{"leftKnee", "leftAnkle"},
	{"nose", "rightShoulder"},
	{"rightShoulder", "rightElbow"},
	{"rightElbow", "rightWrist"},
	{"rightShoulder", "rightHip"},
	{"rightHip", "rightKnee"},
	{"rightKnee", "rightAnkle"},
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

var partIds = map[string]int{
	"nose":          0,
	"leftEye":       1,
	"rightEye":      2,
	"leftEar":       3,
	"rightEar":      4,
	"leftShoulder":  5,
	"rightShoulder": 6,
	"leftElbow":     7,
	"rightElbow":    8,
	"leftWrist":     9,
	"rightWrist":    10,
	"leftHip":       11,
	"rightHip":      12,
	"leftKnee":      13,
	"rightKnee":     14,
	"leftAnkle":     15,
	"rightAnkle":    16,
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

func round(f float64) float64 {
	return float64(math.Round(float64(f)))
}

func clamp(a, min, max float64) float64 {
	if a < min {
		return min
	}
	if a > max {
		return max
	}
	return a
}

func clampint(a, min, max int) int {
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
		y: clamp(point.y/float64(outputStride), 0, float64(height-1)),
		x: clamp(point.x/float64(outputStride), 0, float64(width-1)),
	}
}

func getDisplacement(edgeId int, point vector2d, displacements *tflite.Tensor) vector2d {
	numEdges := displacements.Dim(3) / 2
	return vector2d{
		y: float64(displacements.Float32At(0, int(point.y), int(point.x), edgeId)),
		x: float64(displacements.Float32At(0, int(point.y), int(point.x), edgeId+numEdges)),
	}
}

func roundint(f float64) int {
	return int(math.Round(float64(f)))
}

func traverseToTargetKeypoint(edgeId int, sourceKeypoint *keypoint, targetKeypointId int, scores, offsets *tflite.Tensor, outputStride int, displacements *tflite.Tensor) *keypoint {
	height := scores.Dim(1)
	width := scores.Dim(2)
	// Nearest neighbor interpolation for the source->target displacements.
	sourceKeypointIndices := getStridedIndexNearPoint(sourceKeypoint.position, outputStride, height, width)
	displacement := getDisplacement(edgeId, sourceKeypointIndices, displacements)
	var displacedPoint = addVectors(sourceKeypoint.position, displacement)
	var displacedPointIndices = getStridedIndexNearPoint(displacedPoint, outputStride, height, width)
	var offsetPoint = getOffsetPoint(roundint(displacedPointIndices.y), roundint(displacedPointIndices.x), targetKeypointId, offsets)
	var score = float64(scores.Float32At(0, roundint(displacedPointIndices.y), roundint(displacedPointIndices.x), targetKeypointId))
	displacedPointIndices.scale(float64(outputStride), float64(outputStride))
	var targetKeypoint = addVectors(displacedPointIndices, offsetPoint)
	return &keypoint{
		position: targetKeypoint,
		part:     partNames[targetKeypointId],
		score:    score,
	}
}

func decodePose(
	root *partWithScore,
	scores *tflite.Tensor,
	offsets *tflite.Tensor,
	outputStride int,
	displacementsFwd *tflite.Tensor,
	displacementsBwd *tflite.Tensor) []*keypoint {

	numParts := scores.Dim(3)
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
	scores *tflite.Tensor,
	offsets *tflite.Tensor,
	displacementsFwd *tflite.Tensor,
	displacementsBwd *tflite.Tensor,
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

func getValidResolution(imageScaleFactor float64, inputDimension int, outputStride int) int {
	evenResolution := int(float64(inputDimension)*imageScaleFactor - 1)
	return evenResolution - evenResolution%outputStride + 1
}

func sigmoidTensor(t *tflite.Tensor) {
	fs := t.Float32s()
	for i, f := range fs {
		fs[i] = sigmoid(f)
	}
}

func estimateMultiplePoses(
	interpreter *tflite.Interpreter,
	img image.Image, imageScaleFactor float64, flipHorizontal bool,
	outputStride int, maxDetections int, scoreThreshold float64,
	nmsRadius int) []pose {

	height := img.Bounds().Dy()
	width := img.Bounds().Dx()

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)

	resizedHeight := getValidResolution(imageScaleFactor, height, outputStride)
	resizedWidth := getValidResolution(imageScaleFactor, width, outputStride)

	resized := resize.Resize(uint(resizedWidth), uint(resizedHeight), img, resize.Bilinear)
	ff := input.Float32s()
	for y := 0; y < wanted_height; y++ {
		for x := 0; x < wanted_width; x++ {
			col := resized.At(x, y)
			r, g, b, _ := col.RGBA()
			ff[(y*wanted_width+x)*3+0] = (float32(r)/256 - 127.5) / 127.5
			ff[(y*wanted_width+x)*3+1] = (float32(g)/256 - 127.5) / 127.5
			ff[(y*wanted_width+x)*3+2] = (float32(b)/256 - 127.5) / 127.5
		}
	}

	status := interpreter.Invoke()
	if status != tflite.OK {
		log.Fatal("invoke failed")
	}
	scores := interpreter.GetOutputTensor(0)
	offsets := interpreter.GetOutputTensor(1)
	displacementFwd := interpreter.GetOutputTensor(2)
	displacementBwd := interpreter.GetOutputTensor(3)
	sigmoidTensor(scores)

	poses := decodeMultiplePoses(
		scores, offsets, displacementFwd, displacementBwd, outputStride,
		maxDetections, scoreThreshold, nmsRadius)

	scaleY := float64(img.Bounds().Dy()) / float64(resizedHeight)
	scaleX := float64(img.Bounds().Dx()) / float64(resizedWidth)
	for i := 0; i < len(poses); i++ {
		poses[i].scale(scaleX, scaleY)
	}
	return poses
}
