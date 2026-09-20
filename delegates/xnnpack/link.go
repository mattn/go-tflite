//go:build !xnnpack_builtin

package xnnpack

// Link the XNNPACK delegate as separate shared libraries, which is what the
// bazel build (and the go-tflite buildkit) produces. When TensorFlow Lite is
// built with cmake the delegate is compiled into libtensorflowlite_c.so
// instead; build with -tags xnnpack_builtin to skip these libraries.

/*
#cgo LDFLAGS: -ltensorflowlite-delegate_xnnpack -lXNNPACK
*/
import "C"
