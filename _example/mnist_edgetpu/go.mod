module github.com/mattn/go-tflite/_example/mnist_edgetpu

go 1.13

replace github.com/mattn/go-tflite => ../..

replace github.com/mattn/go-tflite/delegates/edgetpu => ../../delegates/edgetpu

require (
	github.com/mattn/go-tflite v0.0.0-00010101000000-000000000000
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
)
