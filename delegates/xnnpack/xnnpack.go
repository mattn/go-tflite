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

	// WeightsCache shares packed weights between the delegates of multiple
	// interpreter instances running the same model. Optional.
	WeightsCache *WeightsCache
}

// WeightsCache is a cache for packed weights that can be shared between
// multiple delegate instances of the same model. After creating all
// interpreters that share the cache, finalize it with FinalizeHard (fixed
// number of instances, lowest memory) or FinalizeSoft (more instances may
// be added later); inference fails until the cache is finalized. The cache
// must outlive every delegate using it.
type WeightsCache struct {
	c *C.struct_TfLiteXNNPackDelegateWeightsCache
}

// NewWeightsCache creates a new weights cache. Prefer NewWeightsCacheWithSize
// which can reduce memory bandwidth.
func NewWeightsCache() *WeightsCache {
	c := C.TfLiteXNNPackDelegateWeightsCacheCreate()
	if c == nil {
		return nil
	}
	return &WeightsCache{c: c}
}

// NewWeightsCacheWithSize creates a new weights cache that can hold up to
// size bytes without growing.
func NewWeightsCacheWithSize(size int) *WeightsCache {
	c := C.TfLiteXNNPackDelegateWeightsCacheCreateWithSize(C.size_t(size))
	if c == nil {
		return nil
	}
	return &WeightsCache{c: c}
}

// FinalizeSoft finalizes the cache leaving room so that delegates created
// afterwards can still use it when their weights hit the cache.
func (w *WeightsCache) FinalizeSoft() bool {
	return bool(C.TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(w.c))
}

// FinalizeHard freezes the cache and resizes it to the smallest possible
// memory; no delegate can be added afterwards.
func (w *WeightsCache) FinalizeHard() bool {
	return bool(C.TfLiteXNNPackDelegateWeightsCacheFinalizeHard(w.c))
}

// Delete the weights cache. Only call this after every delegate using the
// cache has been deleted.
func (w *WeightsCache) Delete() {
	C.TfLiteXNNPackDelegateWeightsCacheDelete(w.c)
}

// Delegate is the tflite delegate
type Delegate struct {
	d *C.TfLiteDelegate
}

func New(options DelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	coptions := C.TfLiteXNNPackDelegateOptionsDefault()
	coptions.num_threads = C.int32_t(options.NumThreads)
	if options.WeightsCache != nil {
		coptions.weights_cache = options.WeightsCache.c
	}
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
