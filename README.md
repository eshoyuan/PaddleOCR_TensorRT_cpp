English | [简体中文](README_ch.md)


# Introduction

Though PaddleOCR provides support for TensorRT, it is difficult to decouple. This project provides simple code and demonstrates how to use the TensorRT C++ API and ONNX to deploy PaddleOCR text recognition model.

# Enviroment
- Ubuntu 18.04
- CUDA 10.2
- cuDNN 8.4
- OpenCV 3.4.15
- TensorRT 8.4.1.5

# Prerequisites

- Convert the trained model to the inference model.
- Convert the inference model to ONNX model

Details can be found in the PaddleOCR official document.

[Doc-Text Recognition Model Inference](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/inference_en.md#3-text-recognition-model-inference)

[Doc-Paddle2ONNX (Chinese)](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/deploy/paddle2onnx/readme.md#2-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2)

The model used in this tutorial are ch_PP-OCRv2_rec and ch_PP-OCRv3_rec from [PaddleOCR model list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/models_list_en.md#ocr-model-listv3-updated-on-2022428). Download the corresponding inference model and convert it to ONNX model. 

# Usage

- Specify your path in CMakeLists.txt line 14-15.

```bash
# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR "/usr/yyx/tensorrt/TensorRT/")
```

- Modify the parameter in main.cpp line 14-23 if needed.

```cpp
// TODO: Specify your precision here.
options.FP16 = false;
// TODO: Specify your input dimension here.
options.inputDimension = {3,48,320}; // Modify to {3,32,320} when using ppocrv2
// TODO: Specify your character_dict here.
std::string label_path = "../data/ppocr_keys_v1.txt";
// TODO: Specify your test image here.
const std::string inputImage = "../data/word_2.png";
// TODO: Specify your model here.
const std::string onnxModelpath = "../data/modelv3.onnx"; // Modify to "../data/modelv2.onnx" when using ppocrv2
```

- Building the library

```bash
mkdir build
cd build
cmake ..
make
./demo
```

# Results

## PPOCRv2

The result of ch_PP-OCRv2_rec ONNX model on data/word_2.png:

```
yourself        score: 0.95626300573349
```

The result of ch_PP-OCRv2_rec using tools/infer_rec.py in PaddleOCR on data/word_2.png:
```
{"Student": {"label": "yourself", "score": 0.9562630653381348}
 "Teacher": {"label": "yourself", "score": 0.9850824475288391}}
```

## PPOCRv3

The result of ch_PP-OCRv3_rec ONNX model on data/word_2.png:

```
yourself        score: 0.9922693371772766
```

The result of ch_PP-OCRv3_rec using tools/infer_rec.py in PaddleOCR on data/word_2.png:
```
{"Student": {"label": "yourself", "score": 0.9922693371772766}
"Teacher": {"label": "yourself", "score": 0.9903509020805359}}
```

# Thanks to

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) for creating a easy-to-use TensorRT C++ API Tutorial.

- [PaddleOCRv2_TensorRT](https://github.com/zwenyuan1/PaddleOCRv2_TensorRT) for creating some C++ implemention of PaddleOCR preprocess and postprocess method.

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for creating awesome and practical OCR tools that help users train better models and apply them into practice.