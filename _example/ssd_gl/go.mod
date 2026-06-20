module github.com/mattn/go-tflite/_example/ssd_gl

go 1.25.0

replace github.com/mattn/go-tflite => ../..

replace github.com/mattn/go-tflite/delegates/gl => ../../delegates/gl

require (
	github.com/mattn/go-tflite v1.0.5
	gocv.io/x/gocv v0.43.0
	golang.org/x/image v0.43.0
)

require github.com/mattn/go-pointer v0.0.1 // indirect
