package external

/*
#ifndef GO_EXTERNAL_H
#include "external.go.h"
#endif
#cgo CFLAGS: -std=c99
#cgo CXXFLAGS: -std=c99
#cgo LDFLAGS: -ltensorflowlite-delegate_external
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/mattn/go-tflite/delegates"
)

type DelegateOptions struct {
	o       C.TfLiteExternalDelegateOptions
	LibPath string
}

func (o *DelegateOptions) Insert(key, value string) error {
	if C.TfLiteExternalDelegateOptionsInsert(&o.o, C.CString(key), C.CString(value)) == C.kTfLiteError {
		return errors.New("Max options")
	}
	return nil
}

// Delegate is the tflite delegate
type Delegate struct {
	d *C.TfLiteDelegate
}

func New(options DelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	coptions := C.TfLiteExternalDelegateOptionsDefault(C.CString(options.LibPath))
	d = C.TfLiteExternalDelegateCreate(&coptions)
	if d == nil {
		return nil
	}
	return &Delegate{
		d: d,
	}
}

// Delete the delegate
func (d *Delegate) Delete() {
	C.TfLiteExternalDelegateDelete(d.d)
}

// Return a pointer
func (d *Delegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(d.d)
}
