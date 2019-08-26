package cl

/*
#ifndef GO_TFLITE_H
#include "cl.go.h"
#endif
#cgo CFLAGS: -I${SRCDIR}/../../../../../tensorflow/tensorflow
#cgo LDFLAGS: -L${SRCDIR}/../../../../../tensorflow/tensorflow/tensorflow/lite/experimental/c -ltensorflowlite_c -ltensorflowlite_c_delegate_gpu
#cgo linux LDFLAGS: -ldl -lrt
*/
import "C"
import (
	"unsafe"

	"github.com/mattn/go-tflite/delegates"
)

// GpuCompileOptions implement TfLiteGpuCompileOptions.
type GpuCompileOptions struct {
	PrecisionLossAllowed int
	InferencePriority int
}

// GpuDelegateOptions implement TfLiteGpuDelegateOptions.
type GpuDelegateOptions struct {
	GpuCompileOptions GpuCompileOptions
}

// GpuDelegate implement TfLiteGpuCompileOptions.
type GpuDelegate struct {
	d *C.TfLiteDelegate
}

func New(options *GpuDelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	/*
	if options != nil {
		var o C.struct_TfLiteGpuDelegateOptions_New
		o.compile_options.precision_loss_alloed = C.int32_t( options.GpuCompileOptions.PrecisionLossAllowed)
		o.compile_options.inference_priority = C.int32_t( options.GpuCompileOptions.InferencePriority)
		d = C.TfLiteGpuDelegateCreate_New(o)
	} else {
		d = C.TfLiteGpuDelegateCreate_New(nil)
	}
	*/
	d = C.TfLiteGpuDelegateCreate_New(nil)
	if d == nil {
		return nil
	}
	return &GpuDelegate{d: d}
}

func (g *GpuDelegate) Delete() {
	C.TfLiteGpuDelegateDelete_New(g.d)
}

func (g *GpuDelegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(g.d)
}

/*
TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateBindGlBufferToTensor(
    TfLiteDelegate* delegate, GLuint buffer_id, int tensor_index,
    TfLiteType data_type, TfLiteGpuDataLayout data_layout);
TFL_CAPI_EXPORT bool TfLiteGpuDelegateGetSerializedBinaryCache(
    TfLiteDelegate* delegate, size_t* size, const uint8_t** data);
*/
