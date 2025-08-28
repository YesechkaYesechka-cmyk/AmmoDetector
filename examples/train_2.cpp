#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::ml;

class ButtonMatcher {
public:
    struct ButtonFeatures {
        vector<float> shape_features;
        vector<float> color_histogram;
        vector<float> texture_features;
    };

    void train(const vector<string>& image_paths) {
        if (image_paths.empty()) {
            cerr << "Error: No training images provided!" << endl;
            return;
        }

        train_data.clear();
        train_labels.clear();
        button_id_to_name.clear();

        map<string, int> button_name_to_id;
        int current_id = 0;

        for (const auto& path : image_paths) {
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "Warning: Could not load image " << path << " - skipping." << endl;
                continue;
            }

            // Извлечение имени типа из пути (формат: path/type_size.jpg)
            filesystem::path fs_path(path);
            string filename = fs_path.stem().string();

            // Простое извлечение типа (без размера)
            string button_type = filename.substr(0, filename.find_last_of('_'));

            if (button_name_to_id.find(button_type) == button_name_to_id.end()) {
                button_name_to_id[button_type] = current_id;
                button_id_to_name[current_id] = button_type;
                current_id++;
            }

            ButtonFeatures features = extractFeatures(img);
            train_data.push_back(features);
            train_labels.push_back(button_name_to_id[button_type]);

            cout << "Processed: " << path << " -> Type: " << button_type
                 << " (ID: " << button_name_to_id[button_type] << ")" << endl;
        }

        if (train_data.empty()) {
            cerr << "Error: No valid training data collected!" << endl;
            return;
        }

        trainClassifier();
    }

    int match(const Mat& query_img) {
        ButtonFeatures query_features = extractFeatures(query_img);

        if (classifier.empty()) {
            cerr << "Error: Classifier is not trained!" << endl;
            return -1;
        }

        // Подготовка вектора признаков
        vector<float> all_features;
        all_features.insert(all_features.end(), query_features.shape_features.begin(),
                          query_features.shape_features.end());
        all_features.insert(all_features.end(), query_features.color_histogram.begin(),
                          query_features.color_histogram.end());
        all_features.insert(all_features.end(), query_features.texture_features.begin(),
                          query_features.texture_features.end());

        Mat sample(1, all_features.size(), CV_32F);
        for (size_t i = 0; i < all_features.size(); ++i) {
            sample.at<float>(i) = all_features[i];
        }

        return static_cast<int>(classifier->predict(sample));
    }

    string getButtonName(int id) const {
        auto it = button_id_to_name.find(id);
        return it != button_id_to_name.end() ? it->second : "Unknown";
    }

private:
    ButtonFeatures extractFeatures(const Mat& img) {
        ButtonFeatures features;

        // 1. Геометрические признаки
        Mat gray, binary;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 128, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            auto largest_cnt = *max_element(contours.begin(), contours.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) < contourArea(b);
                });

            double area = contourArea(largest_cnt);
            double perimeter = arcLength(largest_cnt, true);
            features.shape_features.push_back(perimeter > 0 ? area / perimeter : 0);

            Rect rect = boundingRect(largest_cnt);
            features.shape_features.push_back(rect.height > 0 ? float(rect.width) / rect.height : 0);
        } else {
            features.shape_features = {0, 0};
        }

        // 2. Цветовые гистограммы
        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);

        int channels[] = {0, 1, 2};
        int histSize[] = {8, 8, 8};
        float h_range[] = {0, 180};
        float sv_range[] = {0, 256};
        const float* ranges[] = {h_range, sv_range, sv_range};

        Mat hist;
        calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
        normalize(hist, hist, 1, 0, NORM_L1);
        hist = hist.reshape(1, 1);
        hist.convertTo(hist, CV_32F);
        features.color_histogram.assign(hist.ptr<float>(), hist.ptr<float>() + hist.cols);

        // 3. Текстура (LBP)
        Mat lbp = computeLBP(gray);
        int lbp_histSize = 256;
        float lbp_range[] = {0, 256};
        const float* lbp_ranges[] = {lbp_range};

        Mat lbp_hist;
        calcHist(&lbp, 1, 0, Mat(), lbp_hist, 1, &lbp_histSize, lbp_ranges, true, false);
        normalize(lbp_hist, lbp_hist, 1, 0, NORM_L1);
        features.texture_features.assign(lbp_hist.ptr<float>(), lbp_hist.ptr<float>() + lbp_hist.rows);

        return features;
    }

    Mat computeLBP(const Mat& gray) {
        Mat lbp = Mat::zeros(gray.size(), CV_8UC1);
        for (int y = 1; y < gray.rows-1; ++y) {
            for (int x = 1; x < gray.cols-1; ++x) {
                uchar center = gray.at<uchar>(y, x);
                uchar code = 0;
                code |= (gray.at<uchar>(y-1, x-1) > center) << 7;
                code |= (gray.at<uchar>(y-1, x) > center) << 6;
                code |= (gray.at<uchar>(y-1, x+1) > center) << 5;
                code |= (gray.at<uchar>(y, x+1) > center) << 4;
                code |= (gray.at<uchar>(y+1, x+1) > center) << 3;
                code |= (gray.at<uchar>(y+1, x) > center) << 2;
                code |= (gray.at<uchar>(y+1, x-1) > center) << 1;
                code |= (gray.at<uchar>(y, x-1) > center) << 0;
                lbp.at<uchar>(y, x) = code;
            }
        }
        return lbp;
    }

    void trainClassifier() {
        int feature_size = train_data[0].shape_features.size() +
                         train_data[0].color_histogram.size() +
                         train_data[0].texture_features.size();

        Mat samples(train_data.size(), feature_size, CV_32F);
        Mat labels(train_labels.size(), 1, CV_32S);

        for (size_t i = 0; i < train_data.size(); ++i) {
            const auto& features = train_data[i];
            float* sample_ptr = samples.ptr<float>(i);

            size_t offset = 0;
            memcpy(sample_ptr + offset, features.shape_features.data(),
                  features.shape_features.size() * sizeof(float));
            offset += features.shape_features.size();

            memcpy(sample_ptr + offset, features.color_histogram.data(),
                  features.color_histogram.size() * sizeof(float));
            offset += features.color_histogram.size();

            memcpy(sample_ptr + offset, features.texture_features.data(),
                  features.texture_features.size() * sizeof(float));

            labels.at<int>(i) = train_labels[i];
        }

        Ptr<RTrees> model = RTrees::create();
        model->setMaxDepth(10);
        model->setMinSampleCount(5);

        TermCriteria term_crit;
        term_crit.maxCount = 100;
        term_crit.epsilon = 0.01f;
        term_crit.type = TermCriteria::MAX_ITER | TermCriteria::EPS;
        model->setTermCriteria(term_crit);

        model->train(samples, ROW_SAMPLE, labels);
        classifier = model;
    }

    vector<ButtonFeatures> train_data;
    vector<int> train_labels;
    Ptr<StatModel> classifier;
    map<int, string> button_id_to_name;
};

float estimateSize(const Mat& img, float ref_size_px, float ref_size_mm) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector<vector<Point>> contours;
    findContours(gray.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return 0.0f;

    Rect rect = boundingRect(contours[0]);
    float size_px = max(rect.width, rect.height);
    return (size_px / ref_size_px) * ref_size_mm;
}

int main() {
    ButtonMatcher matcher;

    // 1. Обучение на примерах пуговиц
    vector<string> train_images = {
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast_2.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug2.jpg",
    };

    cout << "Training classifier..." << endl;
    matcher.train(train_images);
    cout << "Training completed!" << endl << endl;

    // 2. Загрузка тестового изображения
    string test_image_path = "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_etalon.jpg";
    Mat query_img = imread(test_image_path);
    if (query_img.empty()) {
        cerr << "Error: Could not load test image!" << endl;
        return -1;
    }

    // 3. Определение типа пуговицы
    int button_id = matcher.match(query_img);
    string button_name = matcher.getButtonName(button_id);
    cout << "Detected button: " << button_name << " (ID: " << button_id << ")" << endl;

    // 4. Определение размера
    float ref_size_px = 150.0f; // Размер эталона в пикселях
    float ref_size_mm = 24.0f;  // Размер эталона в мм
    float size_mm = estimateSize(query_img, ref_size_px, ref_size_mm);
    cout << "Estimated size: " << size_mm << "mm" << endl;

    // 5. Визуализация
    vector<vector<Point>> contours;
    Mat gray_query_img;
    cvtColor(query_img, gray_query_img, COLOR_BGR2GRAY); // Convert to grayscale
    findContours(gray_query_img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {
        drawContours(query_img, contours, -1, Scalar(0, 255, 0), 2);
        Rect rect = boundingRect(contours[0]);
        rectangle(query_img, rect, Scalar(255, 0, 0), 2);

        putText(query_img,
               format("%s (%.1fmm)", button_name.c_str(), size_mm),
               Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
               Scalar(0, 0, 255), 2);
    }

    imshow("Result", query_img);
    waitKey(0);

    return 0;
}