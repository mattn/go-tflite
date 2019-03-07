# go-tflite

Go binding for TensorFlow Lite

## Usage

See `_example/main.go`

## Requirements

* TensorFlow Lite

## Installation

At the first, you must install Tensorflow Lite C API. The repository of tensorflow must be located on `$GOPATH/src/github.com/tensorflow/tensorflow`.

```
$ cd /path/to/gopath/src/github.com/tensorflow/tensorflow
$ bazel build --config opt --config monolithic tensorflow:libtensorflow_c.so
```

Then go build on go-tflite.

## License

MIT

## Author

Yasuhrio Matsumoto (a.k.a. mattn)
