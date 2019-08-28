package tflite

/*
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#cgo CFLAGS: -I${SRCDIR}/../../tensorflow/tensorflow
#cgo LDFLAGS: -L${SRCDIR}/../../tensorflow/tensorflow/tensorflow/lite/experimental/c -ltensorflowlite_gpu_gl
#cgo pkg-config: egl glesv2
#cgo linux LDFLAGS: -ldl -lrt
*/
import "C"

type GPUDelegate struct {
	o *C.TfLiteDelegate
}

type GPUDelegateOptions struct {
	o *C.TfLiteGpuDelegateOptions
}

func NewGPUDelegateOptionsDefault() *GPUDelegateOptions {
	o := C.TfLiteGpuDelegateOptionsDefault()
	return &GPUDelegateOptions{o: &o}
}

func NewGPUDelegate(options *GPUDelegateOptions) *GPUDelegate {
	if options == nil {
		options = NewGPUDelegateOptionsDefault()
	}
	d := C.TfLiteGpuDelegateCreate(options.o)
	return &GPUDelegate{o: d}
}

func (d *GPUDelegate) d() *C.TfLiteDelegate {
	return d.o
}

func (d *GPUDelegate) Delete() {
	C.TfLiteGpuDelegateDelete(d.d())
}
