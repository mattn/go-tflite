package tflite

/*
#ifndef GO_TFLITE_EXPERIMENTAL_H
#include "tflite_experimental.go.h"
#endif

typedef void* (*f_tflite_registration_init)(TfLiteContext* context, const char* buffer, size_t length);
void* _tflite_registration_init(TfLiteContext* context, char* buffer, size_t length);

typedef void (*f_tflite_registration_free)(TfLiteContext* context, void* buffer);
void _tflite_registration_free(TfLiteContext* context, void* buffer);

typedef TfLiteStatus (*f_tflite_registration_prepare)(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus _tflite_registration_prepare(TfLiteContext* context, TfLiteNode* node);

typedef TfLiteStatus (*f_tflite_registration_invoke)(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus _tflite_registration_invoke(TfLiteContext* context, TfLiteNode* node);

typedef const char* (*f_tflite_registration_profiling_string)(const TfLiteContext* context, const TfLiteNode* node);
char* _tflite_registration_profiling_string(TfLiteContext* context, TfLiteNode* node);

static TfLiteRegistration*
_make_registration(void* o_init, void* o_free, void* o_prepare, void* o_invoke, void* o_profiling_string) {
  TfLiteRegistration* r = (TfLiteRegistration*)malloc(sizeof(TfLiteRegistration));
  r->init = (f_tflite_registration_init) o_init;
  r->free = (f_tflite_registration_free) o_free;
  r->prepare = (f_tflite_registration_prepare) o_prepare;
  r->invoke = (f_tflite_registration_invoke) o_invoke;
  r->profiling_string = (f_tflite_registration_profiling_string) o_profiling_string;
  return r;
}

static void look_context(TfLiteContext *context) {
  context->tensors;
  TfLiteIntArray *plan = NULL;
  context->GetExecutionPlan(context, &plan);
  if (plan == NULL) return;
  int i;
  for (i = 0; i < plan->size; i++) {
    TfLiteNode *node = NULL;
    TfLiteRegistration *reg = NULL;
    context->GetNodeAndRegistration(context, i, &node, &reg);
    printf("%s\n", reg->custom_name);
  }
}

static void writeToTensorAsVector(TfLiteTensor *tensor, char *bytes, size_t size, int nelem) {
  static TfLiteIntArray dummy;
  TfLiteIntArray* new_shape = (TfLiteIntArray*)malloc(sizeof(dummy) + sizeof(dummy.data[0]) * 1);
  if (new_shape) {
    new_shape->size = 1;
    new_shape->data[0] = nelem;
    memcpy(new_shape->data, tensor->dims->data, tensor->dims->size * sizeof(int));
  }

  // TfLiteTensorDataFree
  if (tensor->allocation_type == kTfLiteDynamic && tensor->data.raw) {
    free(tensor->data.raw);
  }
  tensor->data.raw = NULL;

  if (tensor->dims) free(tensor->dims);
  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* q_params =
        (TfLiteAffineQuantization*)(tensor->quantization.params);
    if (q_params->scale) {
      free(q_params->scale);
      q_params->scale = NULL;
    }
    if (q_params->zero_point) {
      free(q_params->zero_point);
      q_params->zero_point = NULL;
    }
    free(q_params);
  }
  tensor->dims = new_shape;
  tensor->data.raw = bytes;
  tensor->bytes = size;
  tensor->allocation_type = kTfLiteMmapRo;

  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->quantization.params = NULL;
}
*/
import "C"
import (
	"bytes"
	"encoding/binary"
	"io"
	"unsafe"
)

const sizeof_int32_t = 4

// ResetVariableTensors resets variable tensors.
func (i *Interpreter) ResetVariableTensors() Status {
	return Status(C.TfLiteInterpreterResetVariableTensors(i.i))
}

/*
type Registration interface {
}

func (o *InterpreterOptions) AddCustomOp(name string, reg *Registration, minVersion, maxVersion int) {
	ptr := C.CString(name)
	defer C.free(unsafe.Pointer(ptr))
	r := C._make_registration()
	C.TfLiteInterpreterOptionsAddCustomOp(o.o, ptr, r, C.int(minVersion), C.int(maxVersion))
}

type registration struct {
	ccxt *C.TfLiteContext
}

//export _tflite_registration_init
func _tflite_registration_init(ccxt *C.TfLiteContext, buffer *C.char, length C.size_t) unsafe.Pointer {
	println("registration.init")
	C.look_context(ccxt)

		//var executionPlan *TfLiteIntArray
		//status := ccxt.GetExecutionPlan(ccxt, &executionPlan)
		//if status != C.kTfLiteOk {
			//return nil
		//}
		//var registration *C.TfLiteRegistration
		//var node *C.TfLiteNode
		//for i := 0; i < executionPlan.size; i++ {
			//ccxt.GetNodeAndRegistration(ccxt, 0, &node, &registration)
		//}

	println(buffer, length)
	return nil
}

//export _tflite_registration_free
func _tflite_registration_free(ccxt *C.TfLiteContext, buffer unsafe.Pointer) {
	println("registration.free")
}

//export _tflite_registration_prepare
func _tflite_registration_prepare(ccxt *C.TfLiteContext, node *C.TfLiteNode) C.TfLiteStatus {
	println("registration.prepare")
	return C.kTfLiteOk
}

//export _tflite_registration_invoke
func _tflite_registration_invoke(ccxt *C.TfLiteContext, node *C.TfLiteNode) C.TfLiteStatus {
	println("registration.invoke")
	return C.kTfLiteOk
}

//export _tflite_registration_profiling_string
func _tflite_registration_profiling_string(ccxt *C.TfLiteContext, node *C.TfLiteNode) *C.char {
	println("registration.profiling_string")
	return nil
}
*/

// ExtRegistration indicate registration structure.
type ExpRegistration struct {
	Init            unsafe.Pointer
	Free            unsafe.Pointer
	Prepare         unsafe.Pointer
	Invoke          unsafe.Pointer
	ProfilingString unsafe.Pointer
}

type BuiltinOperator int

const (
	BuiltinOperator_ADD                          BuiltinOperator = 0
	BuiltinOperator_AVERAGE_POOL_2D              BuiltinOperator = 1
	BuiltinOperator_CONCATENATION                BuiltinOperator = 2
	BuiltinOperator_CONV_2D                      BuiltinOperator = 3
	BuiltinOperator_DEPTHWISE_CONV_2D            BuiltinOperator = 4
	BuiltinOperator_DEQUANTIZE                   BuiltinOperator = 6
	BuiltinOperator_EMBEDDING_LOOKUP             BuiltinOperator = 7
	BuiltinOperator_FLOOR                        BuiltinOperator = 8
	BuiltinOperator_FULLY_CONNECTED              BuiltinOperator = 9
	BuiltinOperator_HASHTABLE_LOOKUP             BuiltinOperator = 10
	BuiltinOperator_L2_NORMALIZATION             BuiltinOperator = 11
	BuiltinOperator_L2_POOL_2D                   BuiltinOperator = 12
	BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION BuiltinOperator = 13
	BuiltinOperator_LOGISTIC                     BuiltinOperator = 14
	BuiltinOperator_LSH_PROJECTION               BuiltinOperator = 15
	BuiltinOperator_LSTM                         BuiltinOperator = 16
	BuiltinOperator_MAX_POOL_2D                  BuiltinOperator = 17
	BuiltinOperator_MUL                          BuiltinOperator = 18
	BuiltinOperator_RELU                         BuiltinOperator = 19
	BuiltinOperator_RELU_N1_TO_1                 BuiltinOperator = 20
	BuiltinOperator_RELU6                        BuiltinOperator = 21
	BuiltinOperator_RESHAPE                      BuiltinOperator = 22
	BuiltinOperator_RESIZE_BILINEAR              BuiltinOperator = 23
	BuiltinOperator_RNN                          BuiltinOperator = 24
	BuiltinOperator_SOFTMAX                      BuiltinOperator = 25
	BuiltinOperator_SPACE_TO_DEPTH               BuiltinOperator = 26
	BuiltinOperator_SVDF                         BuiltinOperator = 27
	BuiltinOperator_TANH                         BuiltinOperator = 28
	BuiltinOperator_CONCAT_EMBEDDINGS            BuiltinOperator = 29
	BuiltinOperator_SKIP_GRAM                    BuiltinOperator = 30
	BuiltinOperator_CALL                         BuiltinOperator = 31
	BuiltinOperator_CUSTOM                       BuiltinOperator = 32
	BuiltinOperator_EMBEDDING_LOOKUP_SPARSE      BuiltinOperator = 33
	BuiltinOperator_PAD                          BuiltinOperator = 34
	BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN  BuiltinOperator = 35
	BuiltinOperator_GATHER                       BuiltinOperator = 36
	BuiltinOperator_BATCH_TO_SPACE_ND            BuiltinOperator = 37
	BuiltinOperator_SPACE_TO_BATCH_ND            BuiltinOperator = 38
	BuiltinOperator_TRANSPOSE                    BuiltinOperator = 39
	BuiltinOperator_MEAN                         BuiltinOperator = 40
	BuiltinOperator_SUB                          BuiltinOperator = 41
	BuiltinOperator_DIV                          BuiltinOperator = 42
	BuiltinOperator_SQUEEZE                      BuiltinOperator = 43
	BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM BuiltinOperator = 44
	BuiltinOperator_STRIDED_SLICE                BuiltinOperator = 45
	BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN   BuiltinOperator = 46
	BuiltinOperator_EXP                          BuiltinOperator = 47
	BuiltinOperator_TOPK_V2                      BuiltinOperator = 48
	BuiltinOperator_SPLIT                        BuiltinOperator = 49
	BuiltinOperator_LOG_SOFTMAX                  BuiltinOperator = 50
	BuiltinOperator_DELEGATE                     BuiltinOperator = 51
	BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM  BuiltinOperator = 52
	BuiltinOperator_CAST                         BuiltinOperator = 53
	BuiltinOperator_PRELU                        BuiltinOperator = 54
	BuiltinOperator_MAXIMUM                      BuiltinOperator = 55
	BuiltinOperator_ARG_MAX                      BuiltinOperator = 56
	BuiltinOperator_MINIMUM                      BuiltinOperator = 57
	BuiltinOperator_LESS                         BuiltinOperator = 58
	BuiltinOperator_NEG                          BuiltinOperator = 59
	BuiltinOperator_PADV2                        BuiltinOperator = 60
	BuiltinOperator_GREATER                      BuiltinOperator = 61
	BuiltinOperator_GREATER_EQUAL                BuiltinOperator = 62
	BuiltinOperator_LESS_EQUAL                   BuiltinOperator = 63
	BuiltinOperator_SELECT                       BuiltinOperator = 64
	BuiltinOperator_SLICE                        BuiltinOperator = 65
	BuiltinOperator_SIN                          BuiltinOperator = 66
	BuiltinOperator_TRANSPOSE_CONV               BuiltinOperator = 67
	BuiltinOperator_SPARSE_TO_DENSE              BuiltinOperator = 68
	BuiltinOperator_TILE                         BuiltinOperator = 69
	BuiltinOperator_EXPAND_DIMS                  BuiltinOperator = 70
	BuiltinOperator_EQUAL                        BuiltinOperator = 71
	BuiltinOperator_NOT_EQUAL                    BuiltinOperator = 72
	BuiltinOperator_LOG                          BuiltinOperator = 73
	BuiltinOperator_SUM                          BuiltinOperator = 74
	BuiltinOperator_SQRT                         BuiltinOperator = 75
	BuiltinOperator_RSQRT                        BuiltinOperator = 76
	BuiltinOperator_SHAPE                        BuiltinOperator = 77
	BuiltinOperator_POW                          BuiltinOperator = 78
	BuiltinOperator_ARG_MIN                      BuiltinOperator = 79
	BuiltinOperator_FAKE_QUANT                   BuiltinOperator = 80
	BuiltinOperator_REDUCE_PROD                  BuiltinOperator = 81
	BuiltinOperator_REDUCE_MAX                   BuiltinOperator = 82
	BuiltinOperator_PACK                         BuiltinOperator = 83
	BuiltinOperator_LOGICAL_OR                   BuiltinOperator = 84
	BuiltinOperator_ONE_HOT                      BuiltinOperator = 85
	BuiltinOperator_LOGICAL_AND                  BuiltinOperator = 86
	BuiltinOperator_LOGICAL_NOT                  BuiltinOperator = 87
	BuiltinOperator_UNPACK                       BuiltinOperator = 88
	BuiltinOperator_REDUCE_MIN                   BuiltinOperator = 89
	BuiltinOperator_FLOOR_DIV                    BuiltinOperator = 90
	BuiltinOperator_REDUCE_ANY                   BuiltinOperator = 91
	BuiltinOperator_SQUARE                       BuiltinOperator = 92
	BuiltinOperator_ZEROS_LIKE                   BuiltinOperator = 93
	BuiltinOperator_FILL                         BuiltinOperator = 94
	BuiltinOperator_FLOOR_MOD                    BuiltinOperator = 95
	BuiltinOperator_RANGE                        BuiltinOperator = 96
	BuiltinOperator_RESIZE_NEAREST_NEIGHBOR      BuiltinOperator = 97
	BuiltinOperator_LEAKY_RELU                   BuiltinOperator = 98
	BuiltinOperator_SQUARED_DIFFERENCE           BuiltinOperator = 99
	BuiltinOperator_MIRROR_PAD                   BuiltinOperator = 100
	BuiltinOperator_ABS                          BuiltinOperator = 101
	BuiltinOperator_SPLIT_V                      BuiltinOperator = 102
	BuiltinOperator_UNIQUE                       BuiltinOperator = 103
	BuiltinOperator_CEIL                         BuiltinOperator = 104
	BuiltinOperator_REVERSE_V2                   BuiltinOperator = 105
	BuiltinOperator_ADD_N                        BuiltinOperator = 106
	BuiltinOperator_GATHER_ND                    BuiltinOperator = 107
	BuiltinOperator_COS                          BuiltinOperator = 108
	BuiltinOperator_WHERE                        BuiltinOperator = 109
	BuiltinOperator_RANK                         BuiltinOperator = 110
	BuiltinOperator_ELU                          BuiltinOperator = 111
	BuiltinOperator_REVERSE_SEQUENCE             BuiltinOperator = 112
	BuiltinOperator_MATRIX_DIAG                  BuiltinOperator = 113
	BuiltinOperator_QUANTIZE                     BuiltinOperator = 114
	BuiltinOperator_MATRIX_SET_DIAG              BuiltinOperator = 115
	BuiltinOperator_MIN                          BuiltinOperator = BuiltinOperator_ADD
	BuiltinOperator_MAX                          BuiltinOperator = BuiltinOperator_MATRIX_SET_DIAG
)

// ExpAddBuiltinOp add builtin op specified by code and registration. Current implementation is work in progress.
func (o *InterpreterOptions) ExpAddBuiltinOp(op BuiltinOperator, reg *ExpRegistration, minVersion, maxVersion int) {
	r := C._make_registration(
		reg.Init,
		reg.Free,
		reg.Prepare,
		reg.Invoke,
		reg.ProfilingString,
	)
	C.TfLiteInterpreterOptionsAddBuiltinOp(o.o, C.TfLiteBuiltinOperator(op), r, C.int(minVersion), C.int(maxVersion))
}

// ExpAddCustomOp add custom op specified by name and registration. Current implementation is work in progress.
func (o *InterpreterOptions) ExpAddCustomOp(name string, reg *ExpRegistration, minVersion, maxVersion int) {
	ptr := C.CString(name)
	defer C.free(unsafe.Pointer(ptr))
	r := C._make_registration(
		reg.Init,
		reg.Free,
		reg.Prepare,
		reg.Invoke,
		reg.ProfilingString,
	)
	C.TfLiteInterpreterOptionsAddCustomOp(o.o, ptr, r, C.int(minVersion), C.int(maxVersion))
}

// SetUseNNAPI enable or disable the NN API for the interpreter (true to enable).
func (o *InterpreterOptions) SetUseNNAPI(enable bool) {
	C.TfLiteInterpreterOptionsSetUseNNAPI(o.o, C.bool(enable))
}

// DynamicBuffer is buffer hold multiple strings.
type DynamicBuffer struct {
	data   bytes.Buffer
	offset []int
}

// AddString append to the dynamic buffer.
func (d *DynamicBuffer) AddString(s string) {
	b := []byte(s)
	d.data.Write(b)
	if len(d.offset) == 0 {
		d.offset = append(d.offset, len(b))
	} else {
		d.offset = append(d.offset, d.offset[len(d.offset)-1]+len(b))
	}
}

// WriteToTensorAsVector write buffer into the tensor as vector.
func (d *DynamicBuffer) WriteToTensorAsVector(t *Tensor) {
	var out bytes.Buffer

	b := make([]byte, 4)

	// Allocate sufficient memory to tensor buffer.
	num_strings := len(d.offset)

	// Set num of string
	binary.LittleEndian.PutUint32(b, uint32(num_strings))
	out.Write(b)

	if num_strings > 0 {

		// Set offset of strings.
		start := sizeof_int32_t + sizeof_int32_t*(num_strings+1)
		offset := start

		binary.LittleEndian.PutUint32(b, uint32(offset))
		out.Write(b)

		for i := 0; i < len(d.offset); i++ {
			offset := start + d.offset[i]
			binary.LittleEndian.PutUint32(b, uint32(offset))
			out.Write(b)
		}

		// Copy data of strings.
		io.Copy(&out, &d.data)
	}

	b = out.Bytes()
	C.writeToTensorAsVector(t.t, (*C.char)(unsafe.Pointer(&b[0])), C.size_t(len(b)), C.int(len(d.offset)))
}

// GetString returns string in the string buffer.
func (t *Tensor) GetString(index int) string {
	if t.Type() != String {
		return ""
	}
	ptr := uintptr(t.Data())
	count := int(*(*C.int32_t)(unsafe.Pointer(ptr)))
	if index >= count {
		return ""
	}
	offset1 := int(*(*C.int32_t)(unsafe.Pointer(ptr + uintptr(4*(index+1)))))
	offset2 := int(*(*C.int32_t)(unsafe.Pointer(ptr + uintptr(4*(index+2)))))
	return string((*((*[1<<31 - 1]uint8)(unsafe.Pointer(ptr))))[offset1:offset2])
}
