# go-tflite examples

Most examples need a `.tflite` model file which is not a product of this
repository. Those models are not checked in; run the download script first:

```
./download-tflite.sh
```

The script only fetches files that are missing, so it is safe to re-run.

## Where the models come from

| Example | File | Source |
|---------|------|--------|
| blazeface | `face_detection_front.tflite`, `face_detection_back.tflite` | [MediaPipe](https://github.com/google/mediapipe) (models bundled up to [v0.7.6](https://github.com/google/mediapipe/tree/v0.7.6/mediapipe/models)) |
| esrgan | `ESRGAN.tflite` | [captain-pool/esrgan-tf2](https://tfhub.dev/captain-pool/lite-model/esrgan-tf2/1) on TensorFlow Hub (now hosted on Kaggle Models) |
| label_image, webcam | `mobilenet_quant_v1_224.tflite` | [TensorFlow hosted models](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_224_android_quant_2017_11_08.zip) (Mobilenet V1 quant, 2017-11-08) |
| label_image_edgetpu | `mobilenet_v2_1.0_224_quant_edgetpu.tflite` | [Coral canned models](https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite) — see `label_image_edgetpu/fetch_testfiles.sh` |
| label_image_xnnpack | `v3-small_224_0.75_float.tflite` | Mobilenet V3 release, [tensorflow/models research/slim](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) |
| pose | `multi_person_mobilenet_v1_075_float.tflite` | [TensorFlow Lite GPU models](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) |
| qa | `mobilebert_float_20191023.tflite` | [TensorFlow Lite BERT QA example](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_float_20191023.tflite) |
| segment | `deeplabv3_257_mv_gpu.tflite` | [TensorFlow Lite GPU models](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite) |
| smartreply | `smartreply.tflite` | [TensorFlow hosted models](https://storage.googleapis.com/download.tensorflow.org/models/smartreply_1.0_2017_11_01.zip) (Smart Reply 1.0, 2017-11-01) |
| ssd, ssd_gl, ssd_sixel, ssd_xnnpack | `detect.tflite` | [TensorFlow hosted models](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) (COCO SSD Mobilenet V1 quant, 2018-06-29) |
| ssd_edgetpu | `mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite`, `coco_labels.txt` | [Coral canned models](https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite) |
| style_transform | `style_predict_quantized_256.tflite`, `style_transfer_quantized_dynamic.tflite` | [Magenta arbitrary image stylization](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite) (TensorFlow Lite style transfer example) |
| text_classification | `text_classification.tflite` | [TensorFlow Lite text classification example](https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite) |
| yolo | `yolov3-tiny.tflite` | [wics1224/yolov3-android-tflite](https://github.com/wics1224/yolov3-android-tflite) (`yolov3_tiny_pb.tflite`, Keras YOLOv3-tiny converted to tflite) |
| yolo | `yolov4-416-fp32.tflite` | [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) (Android app asset; YOLOv4-tiny with decoded box outputs, run with `-model yolov4-416-fp32.tflite`) |
| audio_classification | `yamnet.tflite` | [YAMNet](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) on TensorFlow Hub; class map from [tensorflow/models](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv); test clip from [AudioSet](https://storage.googleapis.com/audioset/miaow_16k.wav) |
| speech_commands | `conv_actions_frozen.tflite` | [TensorFlow hosted models](https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip); test WAV from [tflite-micro micro_speech testdata](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/testdata) |
| gesture_classification | `model.tflite` | [TensorFlow Lite gesture classification example](https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/gesture_classification/android/model_metadata.tflite) (labels.txt is embedded in the model's metadata) |
| recommendation | `recommendation.tflite`, `sorted_movie_vocab.json` | [TensorFlow Lite recommendation example](https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200823/recommendation_rnn_i10o100.tflite) (MovieLens-based RNN demo model) |
| reinforcement_learning | `planestrike.tflite` | [TensorFlow Lite reinforcement learning example](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning) (PlaneStrike Android app asset) |
| ocr | `east_text_detector.tflite` | [EAST text detector](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/dr/1) on TensorFlow Hub |
| ocr | `text_recognition.tflite` | [tulasiram58827/ocr_tflite](https://github.com/tulasiram58827/ocr_tflite) (`keras_ocr_dr.tflite`; the TF Hub keras-ocr variant has an in-graph Flex CTC decoder that stock TFLite cannot run, this one decodes CTC on the app side) |
| video_classification | `movinet_a0_stream.tflite` | [MoViNet A0 stream](https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1) on TensorFlow Hub; Kinetics-600 labels from the [official example assets](https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification) |
| pose | `movenet_lightning.tflite`, `movenet_multipose.tflite` | [MoveNet](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4) singlepose/[multipose](https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1) on TensorFlow Hub (run with `-model movenet_lightning.tflite` etc.) |

## Models generated in this repository

The following examples ship with models that are produced by the `make.py`
(or notebook) found next to them, so they are checked in:

- fizzbuzz, fizzbuzz_edgetpu — `fizzbuzz_model.tflite`, `fizzbuzz_model_quant_edgetpu.tflite`
- mnist, mnist_edgetpu, mnist_reader — `mnist_model.tflite`
- sin — `sin_model.tflite`
- iris — `iris.tflite`
- xor_embedded — model embedded in the source (see `testdata/xor_model.tflite`)
