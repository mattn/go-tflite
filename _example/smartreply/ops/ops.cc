#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/context.h>
#include <tensorflow/lite/kernels/kernel_util.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/tools/make/downloads/absl/absl/strings/str_cat.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/strings/str_split.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/strings/string_view.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/strings/internal/memutil.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/strings/ascii.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/base/internal/throw_delegate.cc>
#include <tensorflow/lite/tools/make/downloads/absl/absl/base/internal/raw_logging.cc>

namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_EXTRACT_FEATURES();
TfLiteRegistration* Register_NORMALIZE();
TfLiteRegistration* Register_PREDICT();
}
}
}

extern "C" {
TfLiteRegistration* Register_EXTRACT_FEATURES() { return tflite::ops::custom::Register_EXTRACT_FEATURES(); }
TfLiteRegistration* Register_NORMALIZE() { return tflite::ops::custom::Register_NORMALIZE(); }
TfLiteRegistration* Register_PREDICT() { return tflite::ops::custom::Register_PREDICT(); }
}
