#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
  return std::distance(first, std::max_element(first, last));
}
int main()
{
  Options options;
  // TODO: Specify your input dimension here.
  options.inputDimension = {3,48,320}; // Modify to {3,32,320} when using ppocrv2
  // TODO: Specify your character_dict here.
  std::string label_path = "../data/ppocr_keys_v1.txt";
  // TODO: Specify your test image here.
  const std::string inputImage = "../data/word_2.png";
  // TODO: Specify your model here.
  const std::string onnxModelpath = "../data/modelv3.onnx"; // Modify to "../data/modelv2.onnx" when using ppocrv2

  std::vector<std::string> label_list_ = ReadDict(label_path);
  Engine engine(options);

  bool succ = engine.build(onnxModelpath);
  if (!succ)
  {
    throw std::runtime_error("Unable to build TRT engine.");
  }

  succ = engine.loadNetwork();
  if (!succ)
  {
    throw std::runtime_error("Unable to load TRT engine.");
  }

  std::vector<cv::Mat> images;
  images.push_back(engine.preprocessImg(inputImage)); // Batchsize = 1

  // Do inference
  std::vector<std::vector<float>> featureVectors;
  int outsize;
  auto t1 = Clock::now(); // Discard the first inference time as it takes longer

  featureVectors.clear();
  outsize = engine.runInference(images, featureVectors); // featureVectors[0] size: [W/4, 6625], in default character_dict.

  // Postprocess
  std::pair<std::vector<std::string>, double> res;
  std::vector<std::string> str_res;
  int argmax_idx;
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  float max_value = 0.0f;
  int m = 0;

  // predict_shape = (1, 80, 6625) in the default model
  // 6625 = 6623 + 2, the length of character_dict is 6623 and 2 character for blank and space.
  std::vector<int> predict_shape = {1, engine.outputDims.d[1], engine.outputDims.d[2]};
  // CTC decode
  // Reference https://github.com/zwenyuan1/PaddleOCRv2_TensorRT/blob/master/src/rec.cpp
  for (int n = 0; n < predict_shape[1]; n++)
  {
    argmax_idx =
        int(argmax(&featureVectors[0][(m * predict_shape[1] + n) * predict_shape[2]],
                   &featureVectors[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));
    max_value =
        float(*std::max_element(&featureVectors[0][(m * predict_shape[1] + n) * predict_shape[2]],
                                &featureVectors[0][(m * predict_shape[1] + n + 1) * predict_shape[2]]));
    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
    {
      score += max_value;
      count += 1;
      str_res.push_back(label_list_[argmax_idx - 1]);
      // I replace "label_list_[argmax_idx]" with "label_list_[argmax_idx - 1]" in my version, otherwise the result can't match.
      // I think this is decided by "use_space_char = True/False" in PaddleOCR config file.
    }
    last_index = argmax_idx;
  }
  score /= count;

  // Print result
  for (int i = 0; i < str_res.size(); i++)
  {
    std::cout << str_res[i];
  }
  std::cout << "\tscore: " << std::setprecision(16) << score << std::endl;

  res.first = str_res;
  res.second = score;

  auto t2 = Clock::now();
  double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // This time is a little more than actual average inference time.
  std::cout << "Success! Inference time: " << totalTime / static_cast<float>(images.size()) << " ms, for batch size of: " << images.size() << std::endl;
  return 0;
}
