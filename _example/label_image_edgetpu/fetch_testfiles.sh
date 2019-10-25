#!/bin/bash
wget https://dl.google.com/coral/canned_models/coco_labels.txt && mv coco_labels.txt labels.txt 
wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
wget https://github.com/tensorflow/tensorflow/raw/r1.15/tensorflow/examples/label_image/data/grace_hopper.jpg