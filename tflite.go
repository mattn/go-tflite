package tflite

/*
#ifndef GO_TFLITE_H
#include "tflite.go.h"
#endif
#cgo LDFLAGS: -ltensorflowlite_c
#cgo android LDFLAGS: -ldl
#cgo linux,!android LDFLAGS: -ldl -lrt
*/
import "C"
import (
	"reflect"
	"unsafe"

	"github.com/mattn/go-pointer"
	"github.com/mattn/go-tflite/delegates"
)

//go:generate stringer -type TensorType,Status -output type_string.go .

// Model is TfLiteModel.
type Model struct {
	m *C.TfLiteModel
}

// NewModel create new Model from buffer.
func NewModel(model_data []byte) *Model {
	m := C.TfLiteModelCreate(C.CBytes(model_data), C.size_t(len(model_data)))
	if m == nil {
		return nil
	}
	return &Model{m: m}
}

// NewModelFromFile create new Model from file data.
func NewModelFromFile(model_path string) *Model {
	ptr := C.CString(model_path)
	defer C.free(unsafe.Pointer(ptr))

	m := C.TfLiteModelCreateFromFile(ptr)
	if m == nil {
		return nil
	}
	return &Model{m: m}
}

// Delete delete instance of model.
func (m *Model) Delete() {
	if m != nil {
		C.TfLiteModelDelete(m.m)
	}
}

// InterpreterOptions implement TfLiteInterpreterOptions.
type InterpreterOptions struct {
	o *C.TfLiteInterpreterOptions
}

// NewInterpreterOptions create new InterpreterOptions.
func NewInterpreterOptions() *InterpreterOptions {
	o := C.TfLiteInterpreterOptionsCreate()
	if o == nil {
		return nil
	}
	return &InterpreterOptions{o: o}
}

// SetNumThread set number of threads.
func (o *InterpreterOptions) SetNumThread(num_threads int) {
	C.TfLiteInterpreterOptionsSetNumThreads(o.o, C.int32_t(num_threads))
}

// SetErrorRepoter set a function of reporter.
func (o *InterpreterOptions) SetErrorReporter(f func(string, interface{}), user_data interface{}) {
	C._TfLiteInterpreterOptionsSetErrorReporter(o.o, pointer.Save(&callbackInfo{
		user_data: user_data,
		f:         f,
	}))
}

func (o *InterpreterOptions) AddDelegate(d delegates.Delegater) {
	C.TfLiteInterpreterOptionsAddDelegate(o.o, (*C.TfLiteDelegate)(d.Ptr()))
}

// Delete delete instance of InterpreterOptions.
func (o *InterpreterOptions) Delete() {
	if o != nil {
		C.TfLiteInterpreterOptionsDelete(o.o)
	}
}

// Interpreter implement TfLiteInterpreter.
type Interpreter struct {
	i *C.TfLiteInterpreter
}

// NewInterpreter create new Interpreter.
func NewInterpreter(model *Model, options *InterpreterOptions) *Interpreter {
	var o *C.TfLiteInterpreterOptions
	if options != nil {
		o = options.o
	}
	i := C.TfLiteInterpreterCreate(model.m, o)
	if i == nil {
		return nil
	}
	return &Interpreter{i: i}
}

// Delete delete instance of Interpreter.
func (i *Interpreter) Delete() {
	if i != nil {
		C.TfLiteInterpreterDelete(i.i)
	}
}

// Tensor implement TfLiteTensor.
type Tensor struct {
	t *C.TfLiteTensor
}

// GetInputTensorCount return number of input tensors.
func (i *Interpreter) GetInputTensorCount() int {
	return int(C.TfLiteInterpreterGetInputTensorCount(i.i))
}

// GetInputTensor return input tensor specified by index.
func (i *Interpreter) GetInputTensor(index int) *Tensor {
	t := C.TfLiteInterpreterGetInputTensor(i.i, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{t: t}
}

// State implement TfLiteStatus.
type Status int

const (
	OK Status = 0
	Error
)

// ResizeInputTensor resize the tensor specified by index with dims.
func (i *Interpreter) ResizeInputTensor(index int, dims []int32) Status {
	s := C.TfLiteInterpreterResizeInputTensor(i.i, C.int32_t(index), (*C.int32_t)(unsafe.Pointer(&dims[0])), C.int32_t(len(dims)))
	return Status(s)
}

// AllocateTensor allocate tensors for the interpreter.
func (i *Interpreter) AllocateTensors() Status {
	if i != nil {
		s := C.TfLiteInterpreterAllocateTensors(i.i)
		return Status(s)
	}
	return Error
}

// Invoke invoke the task.
func (i *Interpreter) Invoke() Status {
	s := C.TfLiteInterpreterInvoke(i.i)
	return Status(s)
}

// GetOutputTensorCount return number of output tensors.
func (i *Interpreter) GetOutputTensorCount() int {
	return int(C.TfLiteInterpreterGetOutputTensorCount(i.i))
}

// GetOutputTensor return output tensor specified by index.
func (i *Interpreter) GetOutputTensor(index int) *Tensor {
	t := C.TfLiteInterpreterGetOutputTensor(i.i, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{t: t}
}

// TensorType is types of the tensor.
type TensorType int

const (
	NoType    TensorType = 0
	Float32   TensorType = 1
	Int32     TensorType = 2
	UInt8     TensorType = 3
	Int64     TensorType = 4
	String    TensorType = 5
	Bool      TensorType = 6
	Int16     TensorType = 7
	Complex64 TensorType = 8
	Int8      TensorType = 9
)

// Type return TensorType.
func (t *Tensor) Type() TensorType {
	return TensorType(C.TfLiteTensorType(t.t))
}

// NumDims return number of dimensions.
func (t *Tensor) NumDims() int {
	return int(C.TfLiteTensorNumDims(t.t))
}

// Dim return dimension of the element specified by index.
func (t *Tensor) Dim(index int) int {
	return int(C.TfLiteTensorDim(t.t, C.int32_t(index)))
}

// Shape return shape of the tensor.
func (t *Tensor) Shape() []int {
	shape := make([]int, t.NumDims())
	for i := 0; i < t.NumDims(); i++ {
		shape[i] = t.Dim(i)
	}
	return shape
}

// ByteSize return byte size of the tensor.
func (t *Tensor) ByteSize() uint {
	return uint(C.TfLiteTensorByteSize(t.t))
}

// Data return pointer of buffer.
func (t *Tensor) Data() unsafe.Pointer {
	return C.TfLiteTensorData(t.t)
}

// Name return name of the tensor.
func (t *Tensor) Name() string {
	return C.GoString(C.TfLiteTensorName(t.t))
}

// QuantizationParams implement TfLiteQuantizationParams.
type QuantizationParams struct {
	Scale     float64
	ZeroPoint int
}

// QuantizationParams return quantization parameters of the tensor.
func (t *Tensor) QuantizationParams() QuantizationParams {
	q := C.TfLiteTensorQuantizationParams(t.t)
	return QuantizationParams{
		Scale:     float64(q.scale),
		ZeroPoint: int(q.zero_point),
	}
}

// CopyFromBuffer write buffer to the tensor.
func (t *Tensor) CopyFromBuffer(b interface{}) Status {
	return Status(C.TfLiteTensorCopyFromBuffer(t.t, unsafe.Pointer(reflect.ValueOf(b).Pointer()), C.size_t(t.ByteSize())))
}

// CopyToBuffer write buffer from the tensor.
func (t *Tensor) CopyToBuffer(b interface{}) Status {
	return Status(C.TfLiteTensorCopyToBuffer(t.t, unsafe.Pointer(reflect.ValueOf(b).Pointer()), C.size_t(t.ByteSize())))
}
