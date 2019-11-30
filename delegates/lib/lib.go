package lib

/*
#ifndef GO_TFLITE_H
#include "lib.go.h"
#endif

#include <stdio.h>

void reportError(const char * err) {
	printf("error creating TFLiteDelegate: %s\n", err);
}

TfLiteDelegate* createDelegate(void *f) {
	// char **options_keys
	// char **options_values

	TfLiteDelegate* (*tflite_plugin_create_delegate)(char **, char **, size_t, void (*)(const char *));
  	tflite_plugin_create_delegate = (TfLiteDelegate* (*)(char **, char **, size_t, void (*)(const char *)))f;

  	return tflite_plugin_create_delegate(NULL, NULL, 0, &reportError);
}

void destroyDelegate(void *f, TfLiteDelegate *delegate) {
	void (*tflite_plugin_destroy_delegate)(void*);
  	tflite_plugin_destroy_delegate = (void (*)(void*))f;

	tflite_plugin_destroy_delegate((void*)delegate);
}

#cgo LDFLAGS: -L/tensorflow/lite/c
#cgo LDFLAGS: -L/usr/local/lib/tensorflow/lite
*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/coreos/pkg/dlopen"
	"github.com/mattn/go-tflite/delegates"
)

// LibDelegate implement Delegater
type LibDelegate struct {
	d *C.TfLiteDelegate

	libHandle   *dlopen.LibHandle
	destroyFunc unsafe.Pointer
}

func New(libraryPath string) delegates.Delegater {
	libraryPaths := []string{libraryPath}
	h, err := dlopen.GetHandle(libraryPaths)
	if err != nil {
		fmt.Printf(`couldn't get a handle to the delegate library: %v\n`, err)
		return nil
	}

	createFunc, err := h.GetSymbolPointer("tflite_plugin_create_delegate")
	if err != nil {
		fmt.Println(`couldn't get symbol "tflite_plugin_create_delegate":`, err)
		return nil
	}

	destroyFunc, err := h.GetSymbolPointer("tflite_plugin_destroy_delegate")
	if err != nil {
		fmt.Println(`couldn't get symbol "tflite_plugin_destroy_delegate":`, err)
		return nil
	}

	d := C.createDelegate(createFunc)
	if d == nil {
		return nil
	}

	return &LibDelegate{d: d, destroyFunc: destroyFunc, libHandle: h}
}

func (g *LibDelegate) Delete() {
	C.destroyDelegate(g.destroyFunc, g.d)
	g.libHandle.Close()
}

func (g *LibDelegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(g.d)
}
