#!/bin/bash
# Download the .tflite models (and label files) used by the examples.
# These files are not products of this repository; see README.md for
# where each of them comes from.
set -e
cd "$(dirname "$0")"

fetch() { # fetch <dest> <url>
  if [ -f "$1" ]; then
    echo "skip  $1"
    return
  fi
  echo "fetch $1"
  curl -fSL -o "$1" "$2"
}

fetch_zip() { # fetch_zip <dir> <member> <url>
  if [ -f "$1/$2" ]; then
    echo "skip  $1/$2"
    return
  fi
  echo "fetch $1/$2"
  tmp=$(mktemp)
  curl -fSL -o "$tmp" "$3"
  unzip -q -o -d "$1" "$tmp" "$2"
  rm -f "$tmp"
}

# blazeface
fetch blazeface/face_detection_front.tflite "https://github.com/google/mediapipe/raw/v0.7.6/mediapipe/models/face_detection_front.tflite"
fetch blazeface/face_detection_back.tflite "https://github.com/google/mediapipe/raw/v0.7.6/mediapipe/models/face_detection_back.tflite"

# esrgan
fetch esrgan/ESRGAN.tflite "https://tfhub.dev/captain-pool/lite-model/esrgan-tf2/1?lite-format=tflite"

# label_image / webcam
fetch_zip label_image mobilenet_quant_v1_224.tflite "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_224_android_quant_2017_11_08.zip"
[ -f webcam/mobilenet_quant_v1_224.tflite ] || cp label_image/mobilenet_quant_v1_224.tflite webcam/

# pose
fetch pose/multi_person_mobilenet_v1_075_float.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite"

# qa
fetch qa/mobilebert_float_20191023.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_float_20191023.tflite"

# segment
fetch segment/deeplabv3_257_mv_gpu.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite"

# smartreply
fetch_zip smartreply smartreply.tflite "https://storage.googleapis.com/download.tensorflow.org/models/smartreply_1.0_2017_11_01.zip"

# ssd / ssd_gl / ssd_sixel / ssd_xnnpack
fetch_zip ssd detect.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
for d in ssd_gl ssd_sixel ssd_xnnpack; do
  [ -f $d/detect.tflite ] || cp ssd/detect.tflite $d/
done

# ssd_edgetpu
fetch ssd_edgetpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite "https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
fetch ssd_edgetpu/coco_labels.txt "https://dl.google.com/coral/canned_models/coco_labels.txt"

# style_transform
fetch style_transform/style_predict_quantized_256.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite"
fetch style_transform/style_transfer_quantized_dynamic.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite"

# text_classification
fetch text_classification/text_classification.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite"

# yolo (yolov3-tiny and yolov4-tiny)
fetch yolo/yolov3-tiny.tflite "https://github.com/wics1224/yolov3-android-tflite/raw/master/app/src/main/assets/yolov3_tiny_pb.tflite"
fetch yolo/yolov4-416-fp32.tflite "https://github.com/hunglc007/tensorflow-yolov4-tflite/raw/master/android/app/src/main/assets/yolov4-416-fp32.tflite"
