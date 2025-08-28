#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Функция для вычисления гистограммы изображения (в HSV)
Mat calculateHistogram(const Mat &image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Параметры гистограммы
    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    const float *ranges[] = {h_range, s_range};
    int channels[] = {0, 1};

    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}

// Функция для сравнения гистограмм
vector<double> compareImages(const Mat &baseImage, const vector<Mat> &images) {
    Mat baseHist = calculateHistogram(baseImage);
    vector<double> scores;

    for (const auto &img: images) {
        Mat hist = calculateHistogram(img);
        double score = compareHist(baseHist, hist, HISTCMP_CORREL);
        scores.push_back(score);
    }

    return scores;
}

int main() {
    // Загрузка эталонного изображения
    Mat reference = imread("/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house_etalon.png");
    if (reference.empty()) {
        cerr << "Could not open reference image!" << endl;
        return -1;
    }

    // Загрузка изображений для сравнения
    vector<Mat> images;
    vector<string> filenames = {
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house1.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house2.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/houses/house3.jpg"
    };

    for (const auto &name: filenames) {
        Mat img = imread(name);
        if (img.empty()) {
            cerr << "Could not open " << name << endl;
            return -1;
        }
        images.push_back(img);
    }

    // Сравнение изображений
    vector<double> scores = compareImages(reference, images);

    // Находим индекс наиболее похожего изображения
    int bestMatchIndex = max_element(scores.begin(), scores.end()) - scores.begin();
    double bestScore = scores[bestMatchIndex];

    // Вывод результатов
    cout << "Comparison scores:" << endl;
    for (size_t i = 0; i < scores.size(); ++i) {
        cout << filenames[i] << ": " << scores[i] << endl;
    }

    cout << "\nMost similar image is: " << filenames[bestMatchIndex]
            << " with score: " << bestScore << endl;

    // Показать результат (опционально)
    imshow("Reference", reference);
    imshow("Most similar", images[bestMatchIndex]);
    waitKey(0);

    return 0;
}
