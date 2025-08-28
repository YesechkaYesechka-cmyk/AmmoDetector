#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // Для SURF
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

// Функция извлечения SURF-признаков
void extractSURFFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    // Параметры можно настроить под конкретную задачу
    int minHessian = 400;
    Ptr<SURF> surf = SURF::create(minHessian);
    surf->detectAndCompute(image, noArray(), keypoints, descriptors);
}

double compareImages(const Mat& img1, const Mat& img2) {
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    // Извлекаем SURF-признаки
    extractSURFFeatures(img1, kp1, desc1);
    extractSURFFeatures(img2, kp2, desc2);

    if (kp1.empty() || kp2.empty() || desc1.empty() || desc2.empty()) {
        return 0.0;
    }

    // Используем FLANN-матчер для SURF (лучше подходит для вещественнозначных дескрипторов)
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Фильтруем совпадения по расстоянию
    double min_dist = 100;
    for (const auto& m : matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }

    vector<DMatch> good_matches;
    for (const auto& m : matches) {
        if (m.distance < max(3 * min_dist, 0.02)) {
            good_matches.push_back(m);
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
    // Проверяем доступность SURF
    // cout << "SURF available: " << haveOpenXFeatures() << endl;

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
        extractSURFFeatures(img1, kp1, desc1);
        extractSURFFeatures(img2, kp2, desc2);

        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        // Фильтрация совпадений
        double min_dist = 100;
        for (const auto& m : matches) {
            if (m.distance < min_dist) min_dist = m.distance;
        }
        vector<DMatch> good_matches;
        for (const auto& m : matches) {
            if (m.distance < max(3 * min_dist, 0.02)) {
                good_matches.push_back(m);
            }
        }

        // Рисуем хорошие совпадения
        Mat img_matches;
        drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
                   Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imshow("Good SURF Matches", img_matches);
        waitKey(0);
    } else {
        cerr << "Error: No valid images for comparison!" << endl;
        return -1;
    }

    return 0;
}