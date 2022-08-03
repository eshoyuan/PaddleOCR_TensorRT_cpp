[English](README.md) | 简体中文

# 介绍

虽然PaddleOCR提供了TensorRT部署支持, 但是其代码比较复杂, 比较难解耦. 本项目提供了相对简洁的代码, 展示如何使用TensorRT C++ API和ONNX进行PaddleOCR文字识别算法的部署.

# 环境
- CUDA 10.2
- cuDNN 8.4
- OpenCV 3.4.15
- TensorRT 8.4.1.5

# 准备
首先需要将Paddle训练模型导出为Paddle推理模型, 再将推理模型转为ONNX模型, 这些在PaddleOCR官方文档有详细过程.

[官方文档-识别模型转inference模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/inference.md#%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B%E8%BD%ACinference%E6%A8%A1%E5%9E%8B)

[官方文档-Paddle2ONNX](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/deploy/paddle2onnx/readme.md#2-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2)

本项目采用的示例模型是官方模型ch_PP-OCRv2_rec和ch_PP-OCRv3_rec ([官方文档-模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/models_list.md#2-%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B)), 直接下载列表中对应的推理模型并转为ONNX模型.

# 运行示例

- 在CMakeLists.txt中14-19行设置自己路径。

```bash
# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR "/usr/yyx/tensorrt/TensorRT/")
# TODO: Specify the path to cuda root dir
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# TODO: Specify the path to opencv root dir
set(OpenCV_DIR "/home/opencv-3.4.15")
```

- 根据需要在main.cpp中14-21行修改参数.

```cpp
// TODO: Specify your input dimension here.
options.inputDimension = {3,48,320}; // Modify to {3,32,320} when using ppocrv2
// TODO: Specify your character_dict here.
std::string label_path = "../data/ppocr_keys_v1.txt";
// TODO: Specify your test image here.
const std::string inputImage = "../data/word_2.png";
// TODO: Specify your model here.
const std::string onnxModelpath = "../data/modelv3.onnx"; // Modify to "../data/modelv2.onnx" when using ppocrv2
```

- 编译运行

```bash
mkdir build
cd build
cmake ..
make
./demo
```

# 结果

## PPOCRv2

本项目使用ch_PP-OCRv2_rec官方推理模型导出为ONNX后在data/word_2.png上的推理结果:

```
yourself        score: 0.95626300573349
```

ch_PP-OCRv2_rec官方训练模型采用PaddleOCR库中tools/infer_rec.py在data/word_2.png上的推理结果:
```
{"Student": {"label": "yourself", "score": 0.9562630653381348}
 "Teacher": {"label": "yourself", "score": 0.9850824475288391}}
```

## PPOCRv3

本项目使用ch_PP-OCRv3_rec官方推理模型导出为ONNX后在data/word_2.png上的推理结果:
```
yourself        score: 0.9922693371772766
```

ch_PP-OCRv3_rec官方训练模型采用PaddleOCR库中tools/infer_rec.py在data/word_2.png上的推理结果:
```
{"Student": {"label": "yourself", "score": 0.9922693371772766}
"Teacher": {"label": "yourself", "score": 0.9903509020805359}}
```
# 致谢

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) 提供了简洁的TensorRT C++ API教程.

- [PaddleOCRv2_TensorRT](https://github.com/zwenyuan1/PaddleOCRv2_TensorRT) 提供部分PaddleOCR预处理和后处理方法的C++实现.

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 打造了一套丰富且实用的OCR工具库, 助力开发者训练出更好的模型, 并应用落地.