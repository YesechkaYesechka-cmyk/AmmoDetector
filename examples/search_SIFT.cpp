#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // Для SIFT
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

struct MatchResult {
    string filename;
    double score;
};

// Функция извлечения SIFT-признаков
void extractSIFTFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    // Создаем детектор SIFT (можно настроить параметры)
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(image, noArray(), keypoints, descriptors);
}

double compareImages(const Mat& img1, const Mat& img2) {
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    // Извлекаем SIFT-признаки
    extractSIFTFeatures(img1, kp1, desc1);
    extractSIFTFeatures(img2, kp2, desc2);

    if (kp1.empty() || kp2.empty() || desc1.empty() || desc2.empty()) {
        return 0.0;
    }

    // Используем FLANN-матчер для SIFT
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2); // Берем 2 ближайших соседа

    // Фильтр по соотношению расстояний (Lowe's ratio test)
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // Оценка схожести на основе хороших совпадений
    double score = 0.0;
    for (const auto& m : good_matches) {
        score += 1.0 / (1.0 + m.distance);
    }
    return good_matches.empty() ? 0.0 : score / good_matches.size();
}

int main() {
    // Проверяем доступность SIFT
    // cout << "SIFT available: " << haveOpenXFeatures() << endl;

    string reference_path = "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_etalon.jpg";

    Mat reference = imread(reference_path, IMREAD_COLOR);
    if (reference.empty()) {
        cerr << "Error: Could not load reference image!" << endl;
        return -1;
    }

    vector<string> compare_paths {
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast_2.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug2.jpg",
    };

    vector<MatchResult> results;

    for (const auto& path : compare_paths) {
        Mat img = imread(path, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Warning: Could not load image " << path << " - skipping." << endl;
            continue;
        }

        double score = compareImages(reference, img);
        results.push_back({path, score});
        cout << "Similarity with " << path << ": " << score << endl;
    }

    if (!results.empty()) {
        auto best_match = *max_element(results.begin(), results.end(),
            [](const MatchResult& a, const MatchResult& b) {
                return a.score < b.score;
            });

        cout << "\nMost similar image: " << best_match.filename
             << " (score: " << best_match.score << ")" << endl;

        // Визуализация совпадений
        Mat img1 = reference, img2 = imread(best_match.filename, IMREAD_COLOR);
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        extractSIFTFeatures(img1, kp1, desc1);
        extractSIFTFeatures(img2, kp2, desc2);

        FlannBasedMatcher matcher;
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);

        // Фильтр по соотношению расстояний
        const float ratio_thresh = 0.7f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        // Рисуем хорошие совпадения
        Mat img_matches;
        drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
                   Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imshow("Good SIFT Matches", img_matches);
        waitKey(0);
    } else {
        cerr << "Error: No valid images for comparison!" << endl;
        return -1;
    }

    return 0;
}