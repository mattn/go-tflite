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

// Version return version string of TensorFlow Lite.
func Version() string {
	return C.GoString(C.TfLiteVersion())
}

// ExtensionApisVersion return version string of the TensorFlow Lite Extension APIs.
func ExtensionApisVersion() string {
	return C.GoString(C.TfLiteExtensionApisVersion())
}

// SchemaVersion return the supported version of the TensorFlow Lite model schema.
func SchemaVersion() int {
	return int(C.TfLiteSchemaVersion())
}

// Model is TfLiteModel.
type Model struct {
	m    *C.TfLiteModel
	data unsafe.Pointer
}

// NewModel create new Model from buffer.
func NewModel(model_data []byte) *Model {
	data := C.CBytes(model_data)
	m := C.TfLiteModelCreate(data, C.size_t(len(model_data)))
	if m == nil {
		C.free(data)
		return nil
	}
	return &Model{m: m, data: data}
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
		if m.data != nil {
			C.free(m.data)
			m.data = nil
		}
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
	if model == nil || model.m == nil {
		return nil
	}
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

// SignatureRunner implement TfLiteSignatureRunner. It is used to run
// inference on a specific SavedModel signature.
type SignatureRunner struct {
	r *C.TfLiteSignatureRunner
}

// GetSignatureCount return the number of signatures defined in the model.
func (i *Interpreter) GetSignatureCount() int {
	return int(C.TfLiteInterpreterGetSignatureCount(i.i))
}

// GetSignatureKey return the key of the signature specified by index.
func (i *Interpreter) GetSignatureKey(index int) string {
	return C.GoString(C.TfLiteInterpreterGetSignatureKey(i.i, C.int32_t(index)))
}

// GetSignatureRunner return the runner for the signature specified by key.
// The returned runner must be deleted with Delete before the interpreter is
// deleted.
func (i *Interpreter) GetSignatureRunner(key string) *SignatureRunner {
	ptr := C.CString(key)
	defer C.free(unsafe.Pointer(ptr))
	r := C.TfLiteInterpreterGetSignatureRunner(i.i, ptr)
	if r == nil {
		return nil
	}
	return &SignatureRunner{r: r}
}

// GetInputCount return number of inputs of the signature.
func (r *SignatureRunner) GetInputCount() int {
	return int(C.TfLiteSignatureRunnerGetInputCount(r.r))
}

// GetInputName return name of the input specified by index.
func (r *SignatureRunner) GetInputName(index int) string {
	return C.GoString(C.TfLiteSignatureRunnerGetInputName(r.r, C.int32_t(index)))
}

// ResizeInputTensor resize the input tensor specified by name with dims.
func (r *SignatureRunner) ResizeInputTensor(name string, dims []int32) Status {
	ptr := C.CString(name)
	defer C.free(unsafe.Pointer(ptr))
	s := C.TfLiteSignatureRunnerResizeInputTensor(r.r, ptr, (*C.int)(unsafe.Pointer(&dims[0])), C.int32_t(len(dims)))
	return Status(s)
}

// AllocateTensors allocate tensors for the signature runner.
func (r *SignatureRunner) AllocateTensors() Status {
	return Status(C.TfLiteSignatureRunnerAllocateTensors(r.r))
}

// GetInputTensor return the input tensor specified by name.
func (r *SignatureRunner) GetInputTensor(name string) *Tensor {
	ptr := C.CString(name)
	defer C.free(unsafe.Pointer(ptr))
	t := C.TfLiteSignatureRunnerGetInputTensor(r.r, ptr)
	if t == nil {
		return nil
	}
	return &Tensor{t: t}
}

// Invoke run the signature.
func (r *SignatureRunner) Invoke() Status {
	return Status(C.TfLiteSignatureRunnerInvoke(r.r))
}

// GetOutputCount return number of outputs of the signature.
func (r *SignatureRunner) GetOutputCount() int {
	return int(C.TfLiteSignatureRunnerGetOutputCount(r.r))
}

// GetOutputName return name of the output specified by index.
func (r *SignatureRunner) GetOutputName(index int) string {
	return C.GoString(C.TfLiteSignatureRunnerGetOutputName(r.r, C.int32_t(index)))
}

// GetOutputTensor return the output tensor specified by name.
func (r *SignatureRunner) GetOutputTensor(name string) *Tensor {
	ptr := C.CString(name)
	defer C.free(unsafe.Pointer(ptr))
	t := C.TfLiteSignatureRunnerGetOutputTensor(r.r, ptr)
	if t == nil {
		return nil
	}
	return &Tensor{t: (*C.TfLiteTensor)(t)}
}

// Delete delete instance of the signature runner.
func (r *SignatureRunner) Delete() {
	if r != nil {
		C.TfLiteSignatureRunnerDelete(r.r)
	}
}

// EnableCancellation enable or disable cancellation for the interpreter
// created with these options. When enabled, an in-flight Invoke call can be
// aborted with Interpreter.Cancel.
func (o *InterpreterOptions) EnableCancellation(enable bool) Status {
	return Status(C.TfLiteInterpreterOptionsEnableCancellation(o.o, C.bool(enable)))
}

// Cancel cancel the in-flight invocation. The Invoke call aborted this way
// returns an error status. Requires EnableCancellation on the options used
// to create the interpreter.
func (i *Interpreter) Cancel() Status {
	return Status(C.TfLiteInterpreterCancel(i.i))
}

// InputTensorIndices return the tensor indices of the input tensors in the
// global tensor list of the interpreter.
func (i *Interpreter) InputTensorIndices() []int {
	p := C.TfLiteInterpreterInputTensorIndices(i.i)
	if p == nil {
		return nil
	}
	indices := make([]int, i.GetInputTensorCount())
	for j := range indices {
		indices[j] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(j)*unsafe.Sizeof(*p))))
	}
	return indices
}

// OutputTensorIndices return the tensor indices of the output tensors in the
// global tensor list of the interpreter.
func (i *Interpreter) OutputTensorIndices() []int {
	p := C.TfLiteInterpreterOutputTensorIndices(i.i)
	if p == nil {
		return nil
	}
	indices := make([]int, i.GetOutputTensorCount())
	for j := range indices {
		indices[j] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(j)*unsafe.Sizeof(*p))))
	}
	return indices
}

// Copy return a copy of the interpreter options.
func (o *InterpreterOptions) Copy() *InterpreterOptions {
	c := C.TfLiteInterpreterOptionsCopy(o.o)
	if c == nil {
		return nil
	}
	return &InterpreterOptions{o: c}
}

// NewModelWithErrorReporter create new Model from buffer. The reporter is
// called with error messages produced while loading the model.
func NewModelWithErrorReporter(model_data []byte, f func(string, interface{}), user_data interface{}) *Model {
	data := C.CBytes(model_data)
	m := C._TfLiteModelCreateWithErrorReporter(data, C.size_t(len(model_data)), pointer.Save(&callbackInfo{
		user_data: user_data,
		f:         f,
	}))
	if m == nil {
		C.free(data)
		return nil
	}
	return &Model{m: m, data: data}
}

// NewModelFromFileWithErrorReporter create new Model from file data. The
// reporter is called with error messages produced while loading the model.
func NewModelFromFileWithErrorReporter(model_path string, f func(string, interface{}), user_data interface{}) *Model {
	ptr := C.CString(model_path)
	defer C.free(unsafe.Pointer(ptr))

	m := C._TfLiteModelCreateFromFileWithErrorReporter(ptr, pointer.Save(&callbackInfo{
		user_data: user_data,
		f:         f,
	}))
	if m == nil {
		return nil
	}
	return &Model{m: m}
}
