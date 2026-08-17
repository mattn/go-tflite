module github.com/mattn/go-tflite/_example/gesture_classification

go 1.21

replace github.com/mattn/go-tflite => ../..

require (
	github.com/mattn/go-tflite v1.0.5
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
)

require github.com/mattn/go-pointer v0.0.1 // indirect
