package xnnpack

/*
#ifndef GO_XNNPACK_H
#include "xnnpack.go.h"
#endif
#cgo LDFLAGS: -ltensorflowlite-delegate_xnnpack -lXNNPACK
*/
import "C"
import (
	"unsafe"

	"github.com/mattn/go-tflite/delegates"
)

type DelegateOptions struct {
	NumThreads int32
}

// Delegate is the tflite delegate
type Delegate struct {
	d *C.TfLiteDelegate
}

func New(options DelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	coptions := C.TfLiteXNNPackDelegateOptionsDefault()
	coptions.num_threads = C.int32_t(options.NumThreads)
	d = C.TfLiteXNNPackDelegateCreate(&coptions)
	if d == nil {
		return nil
	}
	return &Delegate{
		d: d,
	}
}

// Delete the delegate
func (d *Delegate) Delete() {
	C.TfLiteXNNPackDelegateDelete(d.d)
}

// Return a pointer
func (d *Delegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(d.d)
}
