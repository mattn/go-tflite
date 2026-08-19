# go-tflite

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattn/go-tflite/master?labpath=iris.ipynb)

Go binding for TensorFlow Lite

![](https://raw.githubusercontent.com/mattn/go-tflite/master/screenshots/screenshot.png)

## Try it in your browser

Click [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattn/go-tflite/master?labpath=iris.ipynb)
to launch a Jupyter notebook with go-tflite preinstalled and train the iris
classifier right in the browser, no install needed. The same image is
available as `ghcr.io/mattn/go-tflite/gophernotes` (see docker/gophernotes).

## Usage

```go
model := tflite.NewModelFromFile("sin_model.tflite")
if model == nil {
	log.Fatal("cannot load model")
}
defer model.Delete()

options := tflite.NewInterpreterOptions()
defer options.Delete()

interpreter := tflite.NewInterpreter(model, options)
defer interpreter.Delete()

interpreter.AllocateTensors()

v := float64(1.2) * math.Pi / 180.0
input := interpreter.GetInputTensor(0)
input.Float32s()[0] = float32(v)
interpreter.Invoke()
got := float64(interpreter.GetOutputTensor(0).Float32s()[0])
```

See `_example` for more examples

## Requirements

* TensorFlow Lite - This release requires 2.2.0-rc3

## Tensorflow Installation

You must install Tensorflow Lite C API. go-tflite links against
`libtensorflowlite_c.so` only, so there is no need to build the full
TensorFlow library. Assuming the source is under /source/directory/tensorflow

```
$ cd /source/directory/tensorflow
$ bazel build --config opt --config monolithic //tensorflow/lite/c:libtensorflowlite_c.so
```

In order for go to find the headers you must set the CGO_CFLAGS environment variable for the source and libraries of tensorflow.
If your libraries are not installed in a standard location, you must also give the go linker the path to the shared librares
with the CGO_LDFLAGS environment variable.

```
$ export CGO_CFLAGS=-I/source/directory/tensorflow
$ export CGO_LDFLAGS=-L/path/to/tensorflow/libaries
```

If you don't love bazel, you can try `Makefile.tflite`. 
Put this file as `Makefile` in `tensorflow/lite/c`, and run `make`. 
Sorry, this has not been test for Linux or Mac

Then run `go build` on some of the examples.

## Edge TPU
To be able to compile and use the EdgeTPU delegate, you need to install the libraries from here:
https://github.com/google-coral/edgetpu

There is also a deb package here:
https://coral.withgoogle.com/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime

The libraries from should be installed in a system wide library path like `/usr/local/lib`
The include files should be installed somewhere that is accesable from your CGO include path

For x86:
```
cd /tmp && git clone https://github.com/google-coral/edgetpu.git && \
cp edgetpu/libedgetpu/direct/k8/libedgetpu.so.1.0 /usr/local/lib/libedgetpu.so.1.0 && \
ln -rs /usr/local/lib/libedgetpu.so.1.0 /usr/local/lib/libedgetpu.so.1 && \
ln -rs /usr/local/lib/libedgetpu.so.1.0 /usr/local/lib/libedgetpu.so && \
mkdir -p /usr/local/include/libedgetpu && \
cp edgetpu/libedgetpu/edgetpu.h /usr/local/include/edgetpu.h && \
cp edgetpu/libedgetpu/edgetpu_c.h /usr/local/include/edgetpu_c.h && \
rm -Rf edgetpu
```

## Docker build

See <https://github.com/mattn/go-mnist-example/>

## License
MIT

## Author
Yasuhiro Matsumoto (a.k.a. mattn)

