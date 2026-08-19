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
