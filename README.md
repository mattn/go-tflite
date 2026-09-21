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

* TensorFlow Lite C API (`libtensorflowlite_c.so`). CI builds and tests
  against TensorFlow v2.17.1; other recent versions should work as long as
  the C API is compatible.

## Tensorflow Installation

go-tflite links against `libtensorflowlite_c.so` only, so there is no need to
build the full TensorFlow library. There are three ways to get it.

### Prebuilt buildkit

Each release ships `go-tflite-buildkit-<tag>-<target>.tar.gz` for
linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64-msvc and
windows-amd64-mingw, containing the headers and the TensorFlow Lite C API
library.

On Linux and macOS it also contains the XNNPACK delegate libraries. Extract it
into `/usr/local` and you are done:

```
$ curl -fSL -o /tmp/buildkit.tar.gz https://github.com/mattn/go-tflite/releases/download/v1.0.8/go-tflite-buildkit-v1.0.8-linux-amd64.tar.gz
$ sudo tar xzf /tmp/buildkit.tar.gz -C /usr/local
$ sudo ldconfig   # Linux only
```

On Windows it contains `tensorflowlite_c.dll` built with cmake, with XNNPACK
compiled in. The `mingw` variant is built with MinGW-w64 gcc (the toolchain
cgo uses) and ships `libtensorflowlite_c.dll.a`; the `msvc` variant is built
with Visual C++ and ships `tensorflowlite_c.lib`. Either links with cgo.
Extract it anywhere and point cgo at it:

```
> tar xzf go-tflite-buildkit-v1.0.8-windows-amd64-mingw.tar.gz -C C:\tflite
> set CGO_CFLAGS=-IC:/tflite/include
> set CGO_LDFLAGS=-LC:/tflite/lib
> set PATH=C:\tflite\lib;%PATH%
> go build -tags xnnpack_builtin .
```

`ci/build-buildkit.sh` is the script that produces these tarballs, so you can
run it yourself for other TensorFlow versions.

### Build with bazel

```
$ cd /source/directory/tensorflow
$ bazel build -c opt //tensorflow/lite/c:tensorflowlite_c
```

The XNNPACK delegate has no shared library target upstream; see
`ci/build-buildkit.sh` for how to add one.

### Build with cmake

```
$ cmake -S /source/directory/tensorflow/tensorflow/lite/c -B tflite_build
$ cmake --build tflite_build -j
```

This produces `tflite_build/libtensorflowlite_c.so` with XNNPACK compiled in.
Because there is no separate `libtensorflowlite-delegate_xnnpack.so` in this
case, build programs that use `delegates/xnnpack` with
`-tags xnnpack_builtin` so that only `libtensorflowlite_c` is linked.

There is also `Makefile.tflite`, a plain Makefile that builds
`libtensorflowlite_c` when placed in `tensorflow/lite/c`; it is not regularly
tested.

### Environment variables

If the headers and libraries are not installed in a standard location, tell
cgo where to find them:

```
$ export CGO_CFLAGS=-I/source/directory/tensorflow
$ export CGO_LDFLAGS=-L/path/to/tensorflow/libraries
$ export LD_LIBRARY_PATH=/path/to/tensorflow/libraries
```

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

