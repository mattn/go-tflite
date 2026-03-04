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
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))
	cvalue := C.CString(value)
	defer C.free(unsafe.Pointer(cvalue))
	if C.TfLiteExternalDelegateOptionsInsert(&o.o, ckey, cvalue) == C.kTfLiteError {
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
	clibPath := C.CString(options.LibPath)
	defer C.free(unsafe.Pointer(clibPath))
	coptions := C.TfLiteExternalDelegateOptionsDefault(clibPath)
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
