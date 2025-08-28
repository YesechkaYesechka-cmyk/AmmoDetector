#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <tiny_dnn/tiny_dnn.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace tiny_dnn;

const int IMAGE_SIZE = 512;
const int CHANNELS = 1; // grayscale

void prepare_dataset() {
    std::string train_dir = "train";
    std::vector<fs::path> class_names;
    for (const auto& entry : fs::directory_iterator(train_dir)) {
        if (entry.is_directory()) {
            class_names.push_back(entry.path());
        }
    }
    std::sort(class_names.begin(), class_names.end());

    std::vector<vec_t> images;
    std::vector<label_t> labels;

    std::cout << "Found " << class_names.size() << " classes:" << std::endl;
    for (size_t i = 0; i < class_names.size(); ++i) {
        std::cout << "  " << i << ": " << class_names[i].filename() << std::endl;
    }

    for (label_t label = 0; label < class_names.size(); ++label) {
        std::cout << "Processing class: " << class_names[label].filename() << std::endl;
        int count = 0;
        for (const auto& entry : fs::directory_iterator(class_names[label])) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) {
                    std::cerr << "Warning: Could not read image: " << entry.path() << std::endl;
                    continue;
                }

                if (img.cols != IMAGE_SIZE || img.rows != IMAGE_SIZE) {
                    cv::resize(img, img, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
                }

                vec_t v;
                v.reserve(IMAGE_SIZE * IMAGE_SIZE);
                for (int i = 0; i < img.rows; ++i) {
                    for (int j = 0; j < img.cols; ++j) {
                        v.push_back(img.at<uchar>(i, j) / 255.0f);
                    }
                }
                images.push_back(v);
                labels.push_back(label);
                count++;

                if (count % 100 == 0) {
                    std::cout << "Processed " << count << " images for class " << class_names[label].filename() << std::endl;
                }
            }
        }
        std::cout << "Total images for class " << class_names[label].filename() << ": " << count << std::endl;
    }

    std::cout << "Saving dataset with " << images.size() << " images..." << std::endl;
    std::ofstream ofs("dataset.bin", std::ios::binary);
    size_t num_images = images.size();
    size_t num_classes = class_names.size();

    ofs.write(reinterpret_cast<const char*>(&num_images), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&num_classes), sizeof(size_t));

    for (size_t i = 0; i < images.size(); ++i) {
        ofs.write(reinterpret_cast<const char*>(&labels[i]), sizeof(label_t));
        ofs.write(reinterpret_cast<const char*>(images[i].data()), images[i].size() * sizeof(float));
    }
    ofs.close();
    std::cout << "Dataset preparation complete." << std::endl;
}

void train_model(const std::string& model_path) {
    std::ifstream ifs("dataset.bin", std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Run prepare mode first to create dataset.bin" << std::endl;
        return;
    }

    size_t num_images, num_classes;
    ifs.read(reinterpret_cast<char*>(&num_images), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&num_classes), sizeof(size_t));

    std::cout << "Loading dataset with " << num_images << " images and " << num_classes << " classes..." << std::endl;

    std::vector<vec_t> images;
    std::vector<label_t> labels;

    for (size_t i = 0; i < num_images; ++i) {
        label_t label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(label_t));
        vec_t img(IMAGE_SIZE * IMAGE_SIZE);
        ifs.read(reinterpret_cast<char*>(img.data()), IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
        images.push_back(img);
        labels.push_back(label);

        if (i % 100 == 0) {
            std::cout << "Loaded " << i << "/" << num_images << " images..." << std::endl;
        }
    }
    ifs.close();

    std::cout << "Building neural network..." << std::endl;

    // CNN architecture for 512x512 images
    network<sequential> net;
    net << conv(IMAGE_SIZE, IMAGE_SIZE, 7, 1, 32, padding::same) << activation::relu()
        << max_pool(IMAGE_SIZE, IMAGE_SIZE, 32, 2)  // 256x256
        << conv(256, 256, 5, 32, 64, padding::same) << activation::relu()
        << max_pool(256, 256, 64, 2)  // 128x128
        << conv(128, 128, 3, 64, 128, padding::same) << activation::relu()
        << max_pool(128, 128, 128, 2)  // 64x64
        << conv(64, 64, 3, 128, 256, padding::same) << activation::relu()
        << max_pool(64, 64, 256, 2)  // 32x32
        << conv(32, 32, 3, 256, 512, padding::same) << activation::relu()
        << max_pool(32, 32, 512, 2)  // 16x16
        << fully_connected(16 * 16 * 512, 1024) << activation::relu()
        << dropout(1024, 0.5f)
        << fully_connected(1024, static_cast<int>(num_classes)) << activation::softmax();

    std::cout << "Network architecture:" << std::endl;
    std::cout << net << std::endl;

    // Training configuration
    int batch_size = 32;
    int epochs = 50;

    adam optimizer;
    optimizer.alpha = 0.001f;

    std::cout << "Starting training with " << epochs << " epochs..." << std::endl;

    // Use progress display and early stopping
    net.fit<cross_entropy>(optimizer, images, labels, batch_size, epochs,
        []() {},  // on batch complete
        [&]() {   // on epoch complete
            static float best_loss = std::numeric_limits<float>::max();
            float current_loss = net.get_loss<cross_entropy>(images, labels);
            std::cout << "Epoch complete, loss: " << current_loss << std::endl;

            if (current_loss < best_loss) {
                best_loss = current_loss;
                net.save(model_path + ".best");
                std::cout << "Saved best model with loss: " << best_loss << std::endl;
            }
        });

    net.save(model_path);
    std::cout << "Training complete. Model saved to: " << model_path << std::endl;
}

void test_model(const std::string& model_path) {
    std::ifstream ifs("dataset.bin", std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Run prepare mode first to create dataset.bin" << std::endl;
        return;
    }

    size_t num_images, num_classes;
    ifs.read(reinterpret_cast<char*>(&num_images), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&num_classes), sizeof(size_t));
    ifs.close();

    std::cout << "Loading model for " << num_classes << " classes..." << std::endl;
    network<sequential> net;
    net.load(model_path);

    std::string test_dir = "test";
    if (!fs::exists(test_dir)) {
        std::cerr << "Error: Test directory not found: " << test_dir << std::endl;
        return;
    }

    std::cout << "Testing images in directory: " << test_dir << std::endl;

    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Warning: Could not read image: " << entry.path() << std::endl;
                continue;
            }

            if (img.cols != IMAGE_SIZE || img.rows != IMAGE_SIZE) {
                cv::resize(img, img, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
            }

            vec_t v;
            v.reserve(IMAGE_SIZE * IMAGE_SIZE);
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    v.push_back(img.at<uchar>(i, j) / 255.0f);
                }
            }

            auto result = net.predict(v);
            std::cout << "\nImage: " << entry.path().filename() << std::endl;
            std::cout << "Probabilities: ";
            for (size_t i = 0; i < result.size(); ++i) {
                std::cout << "Class " << i << ": " << result[i] * 100.0f << "% ";
            }
            std::cout << std::endl;

            // Find predicted class
            auto max_it = std::max_element(result.begin(), result.end());
            int predicted_class = std::distance(result.begin(), max_it);
            std::cout << "Predicted class: " << predicted_class << " (confidence: "
                      << *max_it * 100.0f << "%)" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [prepare|train|test] [model_path]" << std::endl;
        std::cerr << "  prepare - prepares dataset from train/ directory" << std::endl;
        std::cerr << "  train [model] - trains model (default: model.bin)" << std::endl;
        std::cerr << "  test <model> - tests model on test/ directory" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    try {
        if (mode == "prepare") {
            prepare_dataset();
        } else if (mode == "train") {
            std::string model_path = (argc > 2) ? argv[2] : "model.bin";
            train_model(model_path);
        } else if (mode == "test") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " test <model_path>" << std::endl;
                return 1;
            }
            test_model(argv[2]);
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}