#!/bin/bash
wget https://dl.google.com/coral/canned_models/imagenet_labels.txt && mv imagenet_labels.txt labels.txt
wget https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
wget https://github.com/tensorflow/tensorflow/raw/r1.15/tensorflow/examples/label_image/data/grace_hopper.jpg