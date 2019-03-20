package tflite

/*
#ifndef GO_TFLITE_H
#include "tflite.go.h"
#endif
#cgo CFLAGS: -I${SRCDIR}/../../tensorflow/tensorflow
#cgo LDFLAGS: -L${SRCDIR}/../../tensorflow/tensorflow/tensorflow/lite/experimental/c -ltensorflowlite_c
#cgo linux LDFLAGS: -ldl -lrt
*/
import "C"
import (
	"reflect"
	"unsafe"

	"github.com/mattn/go-pointer"
)

//go:generate stringer -type TensorType,Status -output type_string.go .

type Model struct {
	m *C.TFL_Model
}

func NewModel(model_data []byte) *Model {
	m := C.TFL_NewModel(unsafe.Pointer(&model_data[0]), C.size_t(len(model_data)))
	if m == nil {
		return nil
	}
	return &Model{m: m}
}

func NewModelFromFile(model_path string) *Model {
	ptr := C.CString(model_path)
	defer C.free(unsafe.Pointer(ptr))

	m := C.TFL_NewModelFromFile(ptr)
	if m == nil {
		return nil
	}
	return &Model{m: m}
}

func (m *Model) Delete() {
	C.TFL_DeleteModel(m.m)
}

type InterpreterOptions struct {
	o *C.TFL_InterpreterOptions
}

func NewInterpreterOptions() *InterpreterOptions {
	o := C.TFL_NewInterpreterOptions()
	if o == nil {
		return nil
	}
	return &InterpreterOptions{o: o}
}

func (o *InterpreterOptions) SetNumThread(num_threads int) {
	C.TFL_InterpreterOptionsSetNumThreads(o.o, C.int32_t(num_threads))
}

func (o *InterpreterOptions) SetErrorReporter(f func(string, interface{}), user_data interface{}) {
	C._TFL_InterpreterOptionsSetErrorReporter(o.o, pointer.Save(&callbackInfo{
		user_data: user_data,
		f:         f,
	}))
}

func (o *InterpreterOptions) Delete() {
	C.TFL_DeleteInterpreterOptions(o.o)
}

type Interpreter struct {
	i *C.TFL_Interpreter
}

func NewInterpreter(model *Model, options *InterpreterOptions) *Interpreter {
	var o *C.TFL_InterpreterOptions
	if options != nil {
		o = options.o
	}
	i := C.TFL_NewInterpreter(model.m, o)
	if i == nil {
		return nil
	}
	return &Interpreter{i: i}
}

func (i *Interpreter) Delete() {
	C.TFL_DeleteInterpreter(i.i)
}

type Tensor struct {
	t *C.TFL_Tensor
}

func (i *Interpreter) GetInputTensorCount() int {
	return int(C.TFL_InterpreterGetInputTensorCount(i.i))
}

func (i *Interpreter) GetInputTensor(index int) *Tensor {
	t := C.TFL_InterpreterGetInputTensor(i.i, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{t: t}
}

type Status int

const (
	OK Status = 0
	Error
)

func (i *Interpreter) ResizeInputTensor(index int, dims []int) Status {
	cdims := C.malloc(C.size_t(4 * len(dims)))
	defer C.free(unsafe.Pointer(cdims))
	s := C.TFL_InterpreterResizeInputTensor(i.i, C.int32_t(index), (*C.int)(cdims), C.int32_t(len(dims)))
	return Status(s)
}

func (i *Interpreter) AllocateTensors() Status {
	s := C.TFL_InterpreterAllocateTensors(i.i)
	return Status(s)
}

func (i *Interpreter) Invoke() Status {
	s := C.TFL_InterpreterInvoke(i.i)
	return Status(s)
}

func (i *Interpreter) GetOutputTensorCount() int {
	return int(C.TFL_InterpreterGetOutputTensorCount(i.i))
}

func (i *Interpreter) GetOutputTensor(index int) *Tensor {
	t := C.TFL_InterpreterGetOutputTensor(i.i, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{t: t}
}

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

func (t *Tensor) Type() TensorType {
	return TensorType(C.TFL_TensorType(t.t))
}

func (t *Tensor) NumDims() int {
	return int(C.TFL_TensorNumDims(t.t))
}

func (t *Tensor) Dim(index int) int {
	return int(C.TFL_TensorDim(t.t, C.int32_t(index)))
}

func (t *Tensor) ByteSize() uint {
	return uint(C.TFL_TensorByteSize(t.t))
}

func (t *Tensor) Data() unsafe.Pointer {
	return C.TFL_TensorData(t.t)
}

func (t *Tensor) Name() string {
	return C.GoString(C.TFL_TensorName(t.t))
}

type QuantizationParams struct {
	Scale     float64
	ZeroPoint int
}

func (t *Tensor) QuantizationParams() QuantizationParams {
	q := C.TFL_TensorQuantizationParams(t.t)
	return QuantizationParams{
		Scale:     float64(q.scale),
		ZeroPoint: int(q.zero_point),
	}
}

func (t *Tensor) CopyFromBuffer(b interface{}) Status {
	return Status(C.TFL_TensorCopyFromBuffer(t.t, unsafe.Pointer(reflect.ValueOf(b).Pointer()), C.size_t(t.ByteSize())))
}

func (t *Tensor) CopyToBuffer(b interface{}) Status {
	return Status(C.TFL_TensorCopyToBuffer(t.t, unsafe.Pointer(reflect.ValueOf(b).Pointer()), C.size_t(t.ByteSize())))
}
