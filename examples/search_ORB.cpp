#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

// Структура для хранения результатов сравнения
struct MatchResult {
    string filename;
    double score;
};

// Функция извлечения ORB-признаков
void extractORBFeatures(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) {
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(image, noArray(), keypoints, descriptors);
}

// Функция сравнения двух изображений
double compareImages(const Mat &img1, const Mat &img2) {
    // Извлекаем признаки для обоих изображений
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    extractORBFeatures(img1, kp1, desc1);
    extractORBFeatures(img2, kp2, desc2);

    // Если не найдено ключевых точек или дескрипторов
    if (kp1.empty() || kp2.empty() || desc1.empty() || desc2.empty()) {
        return 0.0;
    }

    // Создаем матчер (Brute-Force с расстоянием Хэмминга)
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Вычисляем оценку схожести
    double score = 0.0;
    for (const auto &m: matches) {
        score += 1.0 / (1.0 + m.distance);
    }
    return matches.empty() ? 0.0 : score / matches.size();
}

int main() {
    // Загружаем эталонное изображение
    string reference_path = "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house_etalon.png";

    Mat reference = imread(reference_path, IMREAD_COLOR);
    if (reference.empty()) {
        cerr << "Error: Could not load reference image!" << endl;
        return -1;
    }

    // Загружаем 3 изображения для сравнения
    vector<string> compare_paths {
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house1.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house2.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house3.jpg"
    };


    vector<MatchResult> results;

    // Сравниваем эталон с каждым изображением
    for (const auto &path: compare_paths) {
        Mat img = imread(path, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Warning: Could not load image " << path << " - skipping." << endl;
            continue;
        }

        double score = compareImages(reference, img);
        results.push_back({path, score});
        cout << "Similarity with " << path << ": " << score << endl;
    }

    // Находим изображение с максимальной схожестью
    if (!results.empty()) {
        auto best_match = *max_element(results.begin(), results.end(),
                                       [](const MatchResult &a, const MatchResult &b) {
                                           return a.score < b.score;
                                       });

        cout << "\nMost similar image: " << best_match.filename
                << " (score: " << best_match.score << ")" << endl;

        // Показываем результат
        Mat reference_resized, best_img = imread(best_match.filename, IMREAD_COLOR);
        resize(reference, reference_resized, best_img.size());
        Mat combined;
        hconcat(reference_resized, best_img, combined);
        imshow("Reference (left) vs Best Match (right)", combined);
        waitKey(0);
    } else {
        cerr << "Error: No valid images for comparison!" << endl;
        return -1;
    }

    return 0;
}
