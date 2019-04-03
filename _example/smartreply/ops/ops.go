package ops

/*
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <tensorflow/lite/experimental/c/c_api.h>
#include <tensorflow/lite/experimental/c/c_api_experimental.h>
#include <tensorflow/lite/context.h>
#cgo windows CFLAGS: -D__LITTLE_ENDIAN__
#cgo CFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow
#cgo CFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
#cgo CFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src
#cgo CFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/absl
#cgo windows CXXFLAGS: -D__LITTLE_ENDIAN__
#cgo CXXFLAGS: -std=c++11
#cgo CXXFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow
#cgo CXXFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
#cgo CXXFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src
#cgo CXXFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/absl
#cgo LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/experimental/c -lre2 -ltensorflow-lite
#cgo LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/experimental/c -ltensorflowlite_c
#cgo windows amd64 LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/gen/windows_x86_64/lib
#cgo linux amd64 LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib
#cgo linux arm32 LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib
#cgo linux LDFLAGS: -ldl -lrt

TfLiteRegistration* Register_EXTRACT_FEATURES();
TfLiteRegistration* Register_NORMALIZE();
TfLiteRegistration* Register_PREDICT();
*/
import "C"
import (
	"reflect"
	"unsafe"

	"github.com/mattn/go-tflite"
)

func wrap(p *C.TfLiteRegistration) *tflite.ExpRegistration {
	return &tflite.ExpRegistration{
		Init:            unsafe.Pointer(reflect.ValueOf(p.init).Pointer()),
		Free:            unsafe.Pointer(reflect.ValueOf(p.free).Pointer()),
		Prepare:         unsafe.Pointer(reflect.ValueOf(p.prepare).Pointer()),
		Invoke:          unsafe.Pointer(reflect.ValueOf(p.invoke).Pointer()),
		ProfilingString: unsafe.Pointer(reflect.ValueOf(p.profiling_string).Pointer()),
	}
}

func Register_EXTRACT_FEATURES() *tflite.ExpRegistration { return wrap(C.Register_EXTRACT_FEATURES()) }
func Register_NORMALIZE() *tflite.ExpRegistration        { return wrap(C.Register_NORMALIZE()) }
func Register_PREDICT() *tflite.ExpRegistration          { return wrap(C.Register_PREDICT()) }
