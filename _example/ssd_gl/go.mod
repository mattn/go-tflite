module github.com/mattn/go-tflite/_example/ssd_gl

go 1.13

replace github.com/mattn/go-tflite => ../..

replace github.com/mattn/go-tflite/delegates/gl => ../../delegates/gl

require (
	github.com/mattn/go-tflite v0.0.0-00010101000000-000000000000
	gocv.io/x/gocv v0.29.0
	golang.org/x/image v0.0.0-20211028202545-6944b10bf410
)
