#include <tensorflow/lite/context.h>

namespace tflite {
namespace ops {
namespace builtin {
TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH_REF();
TfLiteRegistration* Register_LOGISTIC_REF();
TfLiteRegistration* Register_AVERAGE_POOL_REF();
TfLiteRegistration* Register_MAX_POOL_REF();
TfLiteRegistration* Register_L2_POOL_REF();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
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
}
}
}

extern "C" {
TfLiteRegistration* Register_ABS() { return tflite::ops::builtin::Register_ABS(); }
TfLiteRegistration* Register_RELU() { return tflite::ops::builtin::Register_RELU(); }
TfLiteRegistration* Register_RELU_N1_TO_1() { return tflite::ops::builtin::Register_RELU_N1_TO_1(); }
TfLiteRegistration* Register_RELU6() { return tflite::ops::builtin::Register_RELU6(); }
TfLiteRegistration* Register_TANH_REF() { return tflite::ops::builtin::Register_TANH_REF(); }
TfLiteRegistration* Register_LOGISTIC_REF() { return tflite::ops::builtin::Register_LOGISTIC_REF(); }
TfLiteRegistration* Register_AVERAGE_POOL_REF() { return tflite::ops::builtin::Register_AVERAGE_POOL_REF(); }
TfLiteRegistration* Register_MAX_POOL_REF() { return tflite::ops::builtin::Register_MAX_POOL_REF(); }
TfLiteRegistration* Register_L2_POOL_REF() { return tflite::ops::builtin::Register_L2_POOL_REF(); }
TfLiteRegistration* Register_CONV_2D() { return tflite::ops::builtin::Register_CONV_2D(); }
TfLiteRegistration* Register_CONVOLUTION_REF() { return tflite::ops::builtin::Register_CONVOLUTION_REF(); }
TfLiteRegistration* Register_DEPTHWISE_CONV_2D() { return tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(); }
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF() { return tflite::ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF(); }
TfLiteRegistration* Register_SVDF() { return tflite::ops::builtin::Register_SVDF(); }
TfLiteRegistration* Register_RNN() { return tflite::ops::builtin::Register_RNN(); }
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN() { return tflite::ops::builtin::Register_BIDIRECTIONAL_SEQUENCE_RNN(); }
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN() { return tflite::ops::builtin::Register_UNIDIRECTIONAL_SEQUENCE_RNN(); }
TfLiteRegistration* Register_EMBEDDING_LOOKUP() { return tflite::ops::builtin::Register_EMBEDDING_LOOKUP(); }
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE() { return tflite::ops::builtin::Register_EMBEDDING_LOOKUP_SPARSE(); }
TfLiteRegistration* Register_FULLY_CONNECTED_REF() { return tflite::ops::builtin::Register_FULLY_CONNECTED_REF(); }
TfLiteRegistration* Register_LSH_PROJECTION() { return tflite::ops::builtin::Register_LSH_PROJECTION(); }
TfLiteRegistration* Register_HASHTABLE_LOOKUP() { return tflite::ops::builtin::Register_HASHTABLE_LOOKUP(); }
TfLiteRegistration* Register_SOFTMAX() { return tflite::ops::builtin::Register_SOFTMAX(); }
TfLiteRegistration* Register_CONCATENATION_REF() { return tflite::ops::builtin::Register_CONCATENATION_REF(); }
TfLiteRegistration* Register_ADD_REF() { return tflite::ops::builtin::Register_ADD_REF(); }
TfLiteRegistration* Register_SPACE_TO_BATCH_ND_REF() { return tflite::ops::builtin::Register_SPACE_TO_BATCH_ND_REF(); }
TfLiteRegistration* Register_DIV_REF() { return tflite::ops::builtin::Register_DIV_REF(); }
TfLiteRegistration* Register_SUB_REF() { return tflite::ops::builtin::Register_SUB_REF(); }
TfLiteRegistration* Register_BATCH_TO_SPACE_ND_REF() { return tflite::ops::builtin::Register_BATCH_TO_SPACE_ND_REF(); }
TfLiteRegistration* Register_MUL_REF() { return tflite::ops::builtin::Register_MUL_REF(); }
TfLiteRegistration* Register_L2NORM_REF() { return tflite::ops::builtin::Register_L2NORM_REF(); }
TfLiteRegistration* Register_LOCAL_RESPONSE_NORM_REF() { return tflite::ops::builtin::Register_LOCAL_RESPONSE_NORM_REF(); }
TfLiteRegistration* Register_LSTM() { return tflite::ops::builtin::Register_LSTM(); }
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM() { return tflite::ops::builtin::Register_BIDIRECTIONAL_SEQUENCE_LSTM(); }
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM() { return tflite::ops::builtin::Register_UNIDIRECTIONAL_SEQUENCE_LSTM(); }
TfLiteRegistration* Register_PAD_REF() { return tflite::ops::builtin::Register_PAD_REF(); }
TfLiteRegistration* Register_PADV2_REF() { return tflite::ops::builtin::Register_PADV2_REF(); }
TfLiteRegistration* Register_RESHAPE() { return tflite::ops::builtin::Register_RESHAPE(); }
TfLiteRegistration* Register_RESIZE_BILINEAR_REF() { return tflite::ops::builtin::Register_RESIZE_BILINEAR_REF(); }
TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_REF() { return tflite::ops::builtin::Register_RESIZE_NEAREST_NEIGHBOR_REF(); }
TfLiteRegistration* Register_SKIP_GRAM() { return tflite::ops::builtin::Register_SKIP_GRAM(); }
TfLiteRegistration* Register_SPACE_TO_DEPTH_REF() { return tflite::ops::builtin::Register_SPACE_TO_DEPTH_REF(); }
TfLiteRegistration* Register_GATHER() { return tflite::ops::builtin::Register_GATHER(); }
TfLiteRegistration* Register_TRANSPOSE_REF() { return tflite::ops::builtin::Register_TRANSPOSE_REF(); }
TfLiteRegistration* Register_MEAN_REF() { return tflite::ops::builtin::Register_MEAN_REF(); }
TfLiteRegistration* Register_SPLIT() { return tflite::ops::builtin::Register_SPLIT(); }
TfLiteRegistration* Register_SPLIT_V() { return tflite::ops::builtin::Register_SPLIT_V(); }
TfLiteRegistration* Register_SQUEEZE() { return tflite::ops::builtin::Register_SQUEEZE(); }
TfLiteRegistration* Register_STRIDED_SLICE_REF() { return tflite::ops::builtin::Register_STRIDED_SLICE_REF(); }
TfLiteRegistration* Register_EXP() { return tflite::ops::builtin::Register_EXP(); }
TfLiteRegistration* Register_TOPK_V2() { return tflite::ops::builtin::Register_TOPK_V2(); }
TfLiteRegistration* Register_LOG() { return tflite::ops::builtin::Register_LOG(); }
TfLiteRegistration* Register_LOG_SOFTMAX_REF() { return tflite::ops::builtin::Register_LOG_SOFTMAX_REF(); }
TfLiteRegistration* Register_CAST() { return tflite::ops::builtin::Register_CAST(); }
TfLiteRegistration* Register_DEQUANTIZE() { return tflite::ops::builtin::Register_DEQUANTIZE(); }
TfLiteRegistration* Register_PRELU() { return tflite::ops::builtin::Register_PRELU(); }
TfLiteRegistration* Register_MAXIMUM() { return tflite::ops::builtin::Register_MAXIMUM(); }
TfLiteRegistration* Register_MINIMUM() { return tflite::ops::builtin::Register_MINIMUM(); }
TfLiteRegistration* Register_ARG_MAX() { return tflite::ops::builtin::Register_ARG_MAX(); }
TfLiteRegistration* Register_ARG_MIN() { return tflite::ops::builtin::Register_ARG_MIN(); }
TfLiteRegistration* Register_GREATER() { return tflite::ops::builtin::Register_GREATER(); }
TfLiteRegistration* Register_GREATER_EQUAL() { return tflite::ops::builtin::Register_GREATER_EQUAL(); }
TfLiteRegistration* Register_LESS() { return tflite::ops::builtin::Register_LESS(); }
TfLiteRegistration* Register_LESS_EQUAL() { return tflite::ops::builtin::Register_LESS_EQUAL(); }
TfLiteRegistration* Register_FLOOR_REF() { return tflite::ops::builtin::Register_FLOOR_REF(); }
TfLiteRegistration* Register_TILE() { return tflite::ops::builtin::Register_TILE(); }
TfLiteRegistration* Register_NEG() { return tflite::ops::builtin::Register_NEG(); }
TfLiteRegistration* Register_SUM() { return tflite::ops::builtin::Register_SUM(); }
TfLiteRegistration* Register_REDUCE_PROD() { return tflite::ops::builtin::Register_REDUCE_PROD(); }
TfLiteRegistration* Register_REDUCE_MAX() { return tflite::ops::builtin::Register_REDUCE_MAX(); }
TfLiteRegistration* Register_REDUCE_MIN() { return tflite::ops::builtin::Register_REDUCE_MIN(); }
TfLiteRegistration* Register_REDUCE_ANY() { return tflite::ops::builtin::Register_REDUCE_ANY(); }
TfLiteRegistration* Register_SELECT() { return tflite::ops::builtin::Register_SELECT(); }
TfLiteRegistration* Register_SLICE_REF() { return tflite::ops::builtin::Register_SLICE_REF(); }
TfLiteRegistration* Register_SIN() { return tflite::ops::builtin::Register_SIN(); }
TfLiteRegistration* Register_TRANSPOSECONV_REF() { return tflite::ops::builtin::Register_TRANSPOSECONV_REF(); }
TfLiteRegistration* Register_EXPAND_DIMS() { return tflite::ops::builtin::Register_EXPAND_DIMS(); }
TfLiteRegistration* Register_SPARSE_TO_DENSE() { return tflite::ops::builtin::Register_SPARSE_TO_DENSE(); }
TfLiteRegistration* Register_EQUAL() { return tflite::ops::builtin::Register_EQUAL(); }
TfLiteRegistration* Register_NOT_EQUAL() { return tflite::ops::builtin::Register_NOT_EQUAL(); }
TfLiteRegistration* Register_SQRT() { return tflite::ops::builtin::Register_SQRT(); }
TfLiteRegistration* Register_RSQRT() { return tflite::ops::builtin::Register_RSQRT(); }
TfLiteRegistration* Register_SHAPE() { return tflite::ops::builtin::Register_SHAPE(); }
TfLiteRegistration* Register_POW() { return tflite::ops::builtin::Register_POW(); }
TfLiteRegistration* Register_FAKE_QUANT() { return tflite::ops::builtin::Register_FAKE_QUANT(); }
TfLiteRegistration* Register_PACK() { return tflite::ops::builtin::Register_PACK(); }
TfLiteRegistration* Register_ONE_HOT() { return tflite::ops::builtin::Register_ONE_HOT(); }
TfLiteRegistration* Register_LOGICAL_OR() { return tflite::ops::builtin::Register_LOGICAL_OR(); }
TfLiteRegistration* Register_LOGICAL_AND() { return tflite::ops::builtin::Register_LOGICAL_AND(); }
TfLiteRegistration* Register_LOGICAL_NOT() { return tflite::ops::builtin::Register_LOGICAL_NOT(); }
TfLiteRegistration* Register_UNPACK() { return tflite::ops::builtin::Register_UNPACK(); }
TfLiteRegistration* Register_FLOOR_DIV() { return tflite::ops::builtin::Register_FLOOR_DIV(); }
TfLiteRegistration* Register_SQUARE() { return tflite::ops::builtin::Register_SQUARE(); }
TfLiteRegistration* Register_ZEROS_LIKE() { return tflite::ops::builtin::Register_ZEROS_LIKE(); }
TfLiteRegistration* Register_FLOOR_MOD() { return tflite::ops::builtin::Register_FLOOR_MOD(); }
TfLiteRegistration* Register_RANGE() { return tflite::ops::builtin::Register_RANGE(); }
TfLiteRegistration* Register_LEAKY_RELU() { return tflite::ops::builtin::Register_LEAKY_RELU(); }
TfLiteRegistration* Register_SQUARED_DIFFERENCE() { return tflite::ops::builtin::Register_SQUARED_DIFFERENCE(); }
TfLiteRegistration* Register_FILL() { return tflite::ops::builtin::Register_FILL(); }
TfLiteRegistration* Register_MIRROR_PAD() { return tflite::ops::builtin::Register_MIRROR_PAD(); }
}
