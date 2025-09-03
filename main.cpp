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

const int IMAGE_SIZE = 128;
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

    // ДЕТАЛЬНАЯ ПРОВЕРКА ДАННЫХ
    std::cout << "Data verification:" << std::endl;
    std::cout << "Number of images: " << images.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;

    if (!images.empty()) {
        std::cout << "First image size: " << images[0].size() << " elements" << std::endl;
        std::cout << "First image sample values: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << images[0][i] << " ";
        }
        std::cout << std::endl;

        std::cout << "First label: " << labels[0] << std::endl;
    }

    std::cout << "Building neural network..." << std::endl;

    // УЛЬТРА-ПРОСТАЯ АРХИТЕКТУРА - только полносвязные слои
    network<sequential> net;

    // Простая архитектура без сверточных слоев
    net << fully_connected_layer(IMAGE_SIZE * IMAGE_SIZE, 64)
        << activation::relu()
        << fully_connected_layer(64, 32)
        << activation::relu()
        << fully_connected_layer(32, static_cast<size_t>(num_classes))
        << activation::softmax();

    std::cout << "Network architecture: Simple fully connected" << std::endl;

    // Конфигурация обучения
    int batch_size = 4;
    int epochs = 30;  // Еще меньше для теста

    // Убедимся что батч корректен
    if (batch_size > images.size()) {
        batch_size = images.size();
        std::cout << "Adjusted batch size to: " << batch_size << std::endl;
    }

    std::cout << "Dataset size: " << images.size() << " images" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of batches: " << (images.size() + batch_size - 1) / batch_size << std::endl;

    adam optimizer;
    optimizer.alpha = 0.0001f;

    std::cout << "Starting training with " << epochs << " epochs..." << std::endl;

    try {
        // Простая тренировка без callback'ов
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

            // Обучаем всю эпоху
            float loss = net.train<cross_entropy>(optimizer, images, labels, batch_size);

            std::cout << "Epoch " << epoch + 1 << " complete, Loss: " << loss << std::endl;

            // Простая проверка предсказания
            if (!images.empty()) {
                try {
                    auto result = net.predict(images[0]);
                    std::cout << "Sample prediction: ";
                    for (size_t i = 0; i < std::min(result.size(), size_t(3)); ++i) {
                        std::cout << "class" << i << ":" << result[i] << " ";
                    }
                    std::cout << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Prediction test failed: " << e.what() << std::endl;
                }
            }
        }

        net.save(model_path);
        std::cout << "Training complete. Model saved to: " << model_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;

        // Финальная попытка - обучение без батчей
        std::cout << "Trying without batching..." << std::endl;

        network<sequential> final_net;
        final_net << fully_connected_layer(IMAGE_SIZE * IMAGE_SIZE, static_cast<size_t>(num_classes))
                 << activation::softmax();

        // Обучение на одном примере
        if (!images.empty()) {
            std::vector<vec_t> single_image = {images[0]};
            std::vector<label_t> single_label = {labels[0]};

            float loss = final_net.train<cross_entropy>(optimizer, images, labels, batch_size);
            std::cout << "Single sample training loss: " << loss << std::endl;

            final_net.save(model_path + ".final");
            std::cout << "Final model saved." << std::endl;
        }
    }
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