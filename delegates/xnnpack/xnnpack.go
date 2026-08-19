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

// Flags to enable optional features of the delegate. See the Flag constants.
type Flags uint32

const (
	// FlagQS8 enable XNNPACK acceleration for signed quantized 8-bit inference.
	FlagQS8 Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_QS8
	// FlagQU8 enable XNNPACK acceleration for unsigned quantized 8-bit inference.
	FlagQU8 Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_QU8
	// FlagForceFP16 force FP16 inference for FP32 operators.
	FlagForceFP16 Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
	// FlagDynamicFullyConnected enable XNNPACK acceleration for FULLY_CONNECTED
	// operator with dynamic weights.
	FlagDynamicFullyConnected Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED
	// FlagVariableOperators enable XNNPACK acceleration for VAR_HANDLE,
	// READ_VARIABLE, and ASSIGN_VARIABLE operators.
	FlagVariableOperators Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS
	// FlagTransientIndirectionBuffer enable transient indirection buffer to
	// reduce memory usage in selected operators.
	FlagTransientIndirectionBuffer Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER
	// FlagEnableLatestOperators enable the latest XNNPACK operators and features.
	FlagEnableLatestOperators Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS
	// FlagEnableSubgraphReshaping enable XNNPack subgraph reshaping.
	FlagEnableSubgraphReshaping Flags = C.TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING
)

type DelegateOptions struct {
	NumThreads int32

	// Flags override the default feature flags of the delegate when non-zero.
	Flags Flags

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
// with a size large enough for all packed weights: when the cache grows to
// fit more weights, pointers held by the delegates created so far are
// invalidated and inference crashes.
func NewWeightsCache() *WeightsCache {
	c := C.TfLiteXNNPackDelegateWeightsCacheCreate()
	if c == nil {
		return nil
	}
	return &WeightsCache{c: c}
}

// NewWeightsCacheWithSize creates a new weights cache that can hold up to
// size bytes without growing. Make it large enough for all packed weights;
// see NewWeightsCache for what happens when the cache grows.
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
	if options.Flags != 0 {
		coptions.flags = C.uint32_t(options.Flags)
	}
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

// Flags return the feature flags the delegate is using.
func (d *Delegate) Flags() Flags {
	return Flags(C.TfLiteXNNPackDelegateGetFlags(d.d))
}
