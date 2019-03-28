package ops

/*
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <tensorflow/lite/experimental/c/c_api.h>
#include <tensorflow/lite/experimental/c/c_api_experimental.h>
#include <tensorflow/lite/context.h>
#cgo windows CFLAGS: -D__LITTLE_ENDIAN__
#cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow
#cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
#cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src
#cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/absl
#cgo windows CXXFLAGS: -D__LITTLE_ENDIAN__
#cgo CXXFLAGS: -std=c++11
#cgo CXXFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow
#cgo CXXFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
#cgo CXXFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src
#cgo CXXFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/absl
#cgo LDFLAGS: -L${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/experimental/c -lre2 -ltensorflow-lite
#cgo windows amd64 LDFLAGS: -L${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/gen/windows_x86_64/lib
#cgo linux amd64 LDFLAGS: -L${SRCDIR}/../../../tensorflow/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib
#cgo linux LDFLAGS: -ldl -lrt

TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH_REF();
TfLiteRegistration* Register_LOGISTIC_REF();
TfLiteRegistration* Register_AVERAGE_POOL_REF();
TfLiteRegistration* Register_MAX_POOL_REF();
TfLiteRegistration* Register_L2_POOL_REF();
TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_EMBEDDING_LOOKUP();
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE();
TfLiteRegistration* Register_FULLY_CONNECTED_REF();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_CONCATENATION_REF();
TfLiteRegistration* Register_ADD_REF();
TfLiteRegistration* Register_SPACE_TO_BATCH_ND_REF();
TfLiteRegistration* Register_DIV_REF();
TfLiteRegistration* Register_SUB_REF();
TfLiteRegistration* Register_BATCH_TO_SPACE_ND_REF();
TfLiteRegistration* Register_MUL_REF();
TfLiteRegistration* Register_L2NORM_REF();
TfLiteRegistration* Register_LOCAL_RESPONSE_NORM_REF();
TfLiteRegistration* Register_LSTM();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_PAD_REF();
TfLiteRegistration* Register_PADV2_REF();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR_REF();
TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_REF();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SPACE_TO_DEPTH_REF();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_TRANSPOSE_REF();
TfLiteRegistration* Register_MEAN_REF();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SPLIT_V();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE_REF();
TfLiteRegistration* Register_EXP();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOG_SOFTMAX_REF();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_FLOOR_REF();
TfLiteRegistration* Register_TILE();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_SUM();
TfLiteRegistration* Register_REDUCE_PROD();
TfLiteRegistration* Register_REDUCE_MAX();
TfLiteRegistration* Register_REDUCE_MIN();
TfLiteRegistration* Register_REDUCE_ANY();
TfLiteRegistration* Register_SELECT();
TfLiteRegistration* Register_SLICE_REF();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_TRANSPOSECONV_REF();
TfLiteRegistration* Register_EXPAND_DIMS();
TfLiteRegistration* Register_SPARSE_TO_DENSE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SHAPE();
TfLiteRegistration* Register_POW();
TfLiteRegistration* Register_FAKE_QUANT();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_ONE_HOT();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_UNPACK();
TfLiteRegistration* Register_FLOOR_DIV();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_ZEROS_LIKE();
TfLiteRegistration* Register_FLOOR_MOD();
TfLiteRegistration* Register_RANGE();
TfLiteRegistration* Register_LEAKY_RELU();
TfLiteRegistration* Register_SQUARED_DIFFERENCE();
TfLiteRegistration* Register_FILL();
TfLiteRegistration* Register_MIRROR_PAD();
*/
import "C"
import (
	"reflect"
	"unsafe"

	"github.com/mattn/go-tflite"
)

func wrap(p *C.TfLiteRegistration) *tflite.ExpRegistration {
	return &tflite.ExpRegistration{
		Init:            unsafe.Pointer(reflect.ValueOf(p.init).Pointer()),
		Free:            unsafe.Pointer(reflect.ValueOf(p.free).Pointer()),
		Prepare:         unsafe.Pointer(reflect.ValueOf(p.prepare).Pointer()),
		Invoke:          unsafe.Pointer(reflect.ValueOf(p.invoke).Pointer()),
		ProfilingString: unsafe.Pointer(reflect.ValueOf(p.profiling_string).Pointer()),
	}
}

func Register_ABS() *tflite.ExpRegistration { return wrap(C.Register_ABS()) }
func Register_RELU() *tflite.ExpRegistration { return wrap(C.Register_RELU()) }
func Register_RELU_N1_TO_1() *tflite.ExpRegistration { return wrap(C.Register_RELU_N1_TO_1()) }
func Register_RELU6() *tflite.ExpRegistration { return wrap(C.Register_RELU6()) }
func Register_TANH_REF() *tflite.ExpRegistration { return wrap(C.Register_TANH_REF()) }
func Register_LOGISTIC_REF() *tflite.ExpRegistration { return wrap(C.Register_LOGISTIC_REF()) }
func Register_AVERAGE_POOL_REF() *tflite.ExpRegistration { return wrap(C.Register_AVERAGE_POOL_REF()) }
func Register_MAX_POOL_REF() *tflite.ExpRegistration { return wrap(C.Register_MAX_POOL_REF()) }
func Register_L2_POOL_REF() *tflite.ExpRegistration { return wrap(C.Register_L2_POOL_REF()) }
func Register_CONVOLUTION_REF() *tflite.ExpRegistration { return wrap(C.Register_CONVOLUTION_REF()) }
func Register_DEPTHWISE_CONVOLUTION_REF() *tflite.ExpRegistration { return wrap(C.Register_DEPTHWISE_CONVOLUTION_REF()) }
func Register_SVDF() *tflite.ExpRegistration { return wrap(C.Register_SVDF()) }
func Register_RNN() *tflite.ExpRegistration { return wrap(C.Register_RNN()) }
func Register_BIDIRECTIONAL_SEQUENCE_RNN() *tflite.ExpRegistration { return wrap(C.Register_BIDIRECTIONAL_SEQUENCE_RNN()) }
func Register_UNIDIRECTIONAL_SEQUENCE_RNN() *tflite.ExpRegistration { return wrap(C.Register_UNIDIRECTIONAL_SEQUENCE_RNN()) }
func Register_EMBEDDING_LOOKUP() *tflite.ExpRegistration { return wrap(C.Register_EMBEDDING_LOOKUP()) }
func Register_EMBEDDING_LOOKUP_SPARSE() *tflite.ExpRegistration { return wrap(C.Register_EMBEDDING_LOOKUP_SPARSE()) }
func Register_FULLY_CONNECTED_REF() *tflite.ExpRegistration { return wrap(C.Register_FULLY_CONNECTED_REF()) }
func Register_LSH_PROJECTION() *tflite.ExpRegistration { return wrap(C.Register_LSH_PROJECTION()) }
func Register_HASHTABLE_LOOKUP() *tflite.ExpRegistration { return wrap(C.Register_HASHTABLE_LOOKUP()) }
func Register_SOFTMAX() *tflite.ExpRegistration { return wrap(C.Register_SOFTMAX()) }
func Register_CONCATENATION_REF() *tflite.ExpRegistration { return wrap(C.Register_CONCATENATION_REF()) }
func Register_ADD_REF() *tflite.ExpRegistration { return wrap(C.Register_ADD_REF()) }
func Register_SPACE_TO_BATCH_ND_REF() *tflite.ExpRegistration { return wrap(C.Register_SPACE_TO_BATCH_ND_REF()) }
func Register_DIV_REF() *tflite.ExpRegistration { return wrap(C.Register_DIV_REF()) }
func Register_SUB_REF() *tflite.ExpRegistration { return wrap(C.Register_SUB_REF()) }
func Register_BATCH_TO_SPACE_ND_REF() *tflite.ExpRegistration { return wrap(C.Register_BATCH_TO_SPACE_ND_REF()) }
func Register_MUL_REF() *tflite.ExpRegistration { return wrap(C.Register_MUL_REF()) }
func Register_L2NORM_REF() *tflite.ExpRegistration { return wrap(C.Register_L2NORM_REF()) }
func Register_LOCAL_RESPONSE_NORM_REF() *tflite.ExpRegistration { return wrap(C.Register_LOCAL_RESPONSE_NORM_REF()) }
func Register_LSTM() *tflite.ExpRegistration { return wrap(C.Register_LSTM()) }
func Register_BIDIRECTIONAL_SEQUENCE_LSTM() *tflite.ExpRegistration { return wrap(C.Register_BIDIRECTIONAL_SEQUENCE_LSTM()) }
func Register_UNIDIRECTIONAL_SEQUENCE_LSTM() *tflite.ExpRegistration { return wrap(C.Register_UNIDIRECTIONAL_SEQUENCE_LSTM()) }
func Register_PAD_REF() *tflite.ExpRegistration { return wrap(C.Register_PAD_REF()) }
func Register_PADV2_REF() *tflite.ExpRegistration { return wrap(C.Register_PADV2_REF()) }
func Register_RESHAPE() *tflite.ExpRegistration { return wrap(C.Register_RESHAPE()) }
func Register_RESIZE_BILINEAR_REF() *tflite.ExpRegistration { return wrap(C.Register_RESIZE_BILINEAR_REF()) }
func Register_RESIZE_NEAREST_NEIGHBOR_REF() *tflite.ExpRegistration { return wrap(C.Register_RESIZE_NEAREST_NEIGHBOR_REF()) }
func Register_SKIP_GRAM() *tflite.ExpRegistration { return wrap(C.Register_SKIP_GRAM()) }
func Register_SPACE_TO_DEPTH_REF() *tflite.ExpRegistration { return wrap(C.Register_SPACE_TO_DEPTH_REF()) }
func Register_GATHER() *tflite.ExpRegistration { return wrap(C.Register_GATHER()) }
func Register_TRANSPOSE_REF() *tflite.ExpRegistration { return wrap(C.Register_TRANSPOSE_REF()) }
func Register_MEAN_REF() *tflite.ExpRegistration { return wrap(C.Register_MEAN_REF()) }
func Register_SPLIT() *tflite.ExpRegistration { return wrap(C.Register_SPLIT()) }
func Register_SPLIT_V() *tflite.ExpRegistration { return wrap(C.Register_SPLIT_V()) }
func Register_SQUEEZE() *tflite.ExpRegistration { return wrap(C.Register_SQUEEZE()) }
func Register_STRIDED_SLICE_REF() *tflite.ExpRegistration { return wrap(C.Register_STRIDED_SLICE_REF()) }
func Register_EXP() *tflite.ExpRegistration { return wrap(C.Register_EXP()) }
func Register_TOPK_V2() *tflite.ExpRegistration { return wrap(C.Register_TOPK_V2()) }
func Register_LOG() *tflite.ExpRegistration { return wrap(C.Register_LOG()) }
func Register_LOG_SOFTMAX_REF() *tflite.ExpRegistration { return wrap(C.Register_LOG_SOFTMAX_REF()) }
func Register_CAST() *tflite.ExpRegistration { return wrap(C.Register_CAST()) }
func Register_DEQUANTIZE() *tflite.ExpRegistration { return wrap(C.Register_DEQUANTIZE()) }
func Register_PRELU() *tflite.ExpRegistration { return wrap(C.Register_PRELU()) }
func Register_MAXIMUM() *tflite.ExpRegistration { return wrap(C.Register_MAXIMUM()) }
func Register_MINIMUM() *tflite.ExpRegistration { return wrap(C.Register_MINIMUM()) }
func Register_ARG_MAX() *tflite.ExpRegistration { return wrap(C.Register_ARG_MAX()) }
func Register_ARG_MIN() *tflite.ExpRegistration { return wrap(C.Register_ARG_MIN()) }
func Register_GREATER() *tflite.ExpRegistration { return wrap(C.Register_GREATER()) }
func Register_GREATER_EQUAL() *tflite.ExpRegistration { return wrap(C.Register_GREATER_EQUAL()) }
func Register_LESS() *tflite.ExpRegistration { return wrap(C.Register_LESS()) }
func Register_LESS_EQUAL() *tflite.ExpRegistration { return wrap(C.Register_LESS_EQUAL()) }
func Register_FLOOR_REF() *tflite.ExpRegistration { return wrap(C.Register_FLOOR_REF()) }
func Register_TILE() *tflite.ExpRegistration { return wrap(C.Register_TILE()) }
func Register_NEG() *tflite.ExpRegistration { return wrap(C.Register_NEG()) }
func Register_SUM() *tflite.ExpRegistration { return wrap(C.Register_SUM()) }
func Register_REDUCE_PROD() *tflite.ExpRegistration { return wrap(C.Register_REDUCE_PROD()) }
func Register_REDUCE_MAX() *tflite.ExpRegistration { return wrap(C.Register_REDUCE_MAX()) }
func Register_REDUCE_MIN() *tflite.ExpRegistration { return wrap(C.Register_REDUCE_MIN()) }
func Register_REDUCE_ANY() *tflite.ExpRegistration { return wrap(C.Register_REDUCE_ANY()) }
func Register_SELECT() *tflite.ExpRegistration { return wrap(C.Register_SELECT()) }
func Register_SLICE_REF() *tflite.ExpRegistration { return wrap(C.Register_SLICE_REF()) }
func Register_SIN() *tflite.ExpRegistration { return wrap(C.Register_SIN()) }
func Register_TRANSPOSECONV_REF() *tflite.ExpRegistration { return wrap(C.Register_TRANSPOSECONV_REF()) }
func Register_EXPAND_DIMS() *tflite.ExpRegistration { return wrap(C.Register_EXPAND_DIMS()) }
func Register_SPARSE_TO_DENSE() *tflite.ExpRegistration { return wrap(C.Register_SPARSE_TO_DENSE()) }
func Register_EQUAL() *tflite.ExpRegistration { return wrap(C.Register_EQUAL()) }
func Register_NOT_EQUAL() *tflite.ExpRegistration { return wrap(C.Register_NOT_EQUAL()) }
func Register_SQRT() *tflite.ExpRegistration { return wrap(C.Register_SQRT()) }
func Register_RSQRT() *tflite.ExpRegistration { return wrap(C.Register_RSQRT()) }
func Register_SHAPE() *tflite.ExpRegistration { return wrap(C.Register_SHAPE()) }
func Register_POW() *tflite.ExpRegistration { return wrap(C.Register_POW()) }
func Register_FAKE_QUANT() *tflite.ExpRegistration { return wrap(C.Register_FAKE_QUANT()) }
func Register_PACK() *tflite.ExpRegistration { return wrap(C.Register_PACK()) }
func Register_ONE_HOT() *tflite.ExpRegistration { return wrap(C.Register_ONE_HOT()) }
func Register_LOGICAL_OR() *tflite.ExpRegistration { return wrap(C.Register_LOGICAL_OR()) }
func Register_LOGICAL_AND() *tflite.ExpRegistration { return wrap(C.Register_LOGICAL_AND()) }
func Register_LOGICAL_NOT() *tflite.ExpRegistration { return wrap(C.Register_LOGICAL_NOT()) }
func Register_UNPACK() *tflite.ExpRegistration { return wrap(C.Register_UNPACK()) }
func Register_FLOOR_DIV() *tflite.ExpRegistration { return wrap(C.Register_FLOOR_DIV()) }
func Register_SQUARE() *tflite.ExpRegistration { return wrap(C.Register_SQUARE()) }
func Register_ZEROS_LIKE() *tflite.ExpRegistration { return wrap(C.Register_ZEROS_LIKE()) }
func Register_FLOOR_MOD() *tflite.ExpRegistration { return wrap(C.Register_FLOOR_MOD()) }
func Register_RANGE() *tflite.ExpRegistration { return wrap(C.Register_RANGE()) }
func Register_LEAKY_RELU() *tflite.ExpRegistration { return wrap(C.Register_LEAKY_RELU()) }
func Register_SQUARED_DIFFERENCE() *tflite.ExpRegistration { return wrap(C.Register_SQUARED_DIFFERENCE()) }
func Register_FILL() *tflite.ExpRegistration { return wrap(C.Register_FILL()) }
func Register_MIRROR_PAD() *tflite.ExpRegistration { return wrap(C.Register_MIRROR_PAD()) }
