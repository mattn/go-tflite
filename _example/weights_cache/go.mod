module github.com/mattn/go-tflite/_example/weights_cache

go 1.13

replace github.com/mattn/go-tflite => ../..

replace github.com/mattn/go-tflite/delegates/xnnpack => ../../delegates/xnnpack

require github.com/mattn/go-tflite v1.0.5
