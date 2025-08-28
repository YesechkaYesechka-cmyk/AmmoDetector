#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

class ButtonMatcher {
public:
    struct ButtonFeatures {
        vector<float> shape_features;
        vector<float> color_histogram;
        vector<float> texture_features;
    };

    void train(const vector<string> &image_paths) {
        // Проверка на пустой вход
        if (image_paths.empty()) {
            cerr << "Error: No training images provided!" << endl;
            return;
        }

        // Очистка предыдущих данных
        train_data.clear();
        train_labels.clear();

        // Словарь для соответствия имен пуговиц и их ID
        map<string, int> button_name_to_id;
        int current_id = 0;

        for (const auto &path: image_paths) {
            // Загрузка изображения
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "Warning: Could not load image " << path << " - skipping." << endl;
                continue;
            }

            // Извлечение имени пуговицы из пути (пример: "buttons/button1_typeA_10mm.jpg")
            std::filesystem::path fs_path(path);
            string filename = fs_path.stem().string(); // "button1_typeA_10mm"

            // Разбиваем имя файла на части
            vector<string> parts;
            size_t pos = 0;
            while ((pos = filename.find('_')) != string::npos) {
                parts.push_back(filename.substr(0, pos));
                filename.erase(0, pos + 1);
            }
            parts.push_back(filename);

            if (parts.empty()) {
                cerr << "Warning: Invalid filename format for " << path << " - skipping." << endl;
                continue;
            }

            // Определяем тип пуговицы (например, "typeA" из "button1_typeA_10mm")
            string button_type = parts.size() > 1 ? parts[1] : parts[0];

            // Присваиваем ID типу пуговицы
            if (button_name_to_id.find(button_type) == button_name_to_id.end()) {
                button_name_to_id[button_type] = current_id++;
            }

            // Извлекаем признаки
            ButtonFeatures features = extractFeatures(img);

            // Добавляем в обучающую выборку
            train_data.push_back(features);
            train_labels.push_back(button_name_to_id[button_type]);

            // Логирование процесса
            cout << "Processed: " << path << " -> Type: " << button_type
                    << " (ID: " << button_name_to_id[button_type] << ")" << endl;
        }

        // Проверка, что мы собрали достаточно данных
        if (train_data.empty()) {
            cerr << "Error: No valid training data collected!" << endl;
            return;
        }

        // Обучаем классификатор
        trainClassifier();

        // Сохраняем mapping имен в ID для последующего использования
        saveLabelMapping(button_name_to_id);
    }

    void saveLabelMapping(const map<string, int> &mapping) {
        std::ofstream out("button_labels.map");
        for (const auto &pair: mapping) {
            std::cout << pair.first << " " << pair.second << endl;
        }
    }

    map<string, int> loadLabelMapping(const string &path) {
        map<string, int> mapping;
        std::ifstream in(path);
        string name;
        int id;
        while (in >> name >> id) {
            mapping[name] = id;
        }
        return mapping;
    }


    int match(const Mat& query_img) {
        // 1. Извлекаем признаки из входного изображения
        ButtonFeatures query_features = extractFeatures(query_img);

        // 2. Проверяем, что классификатор обучен
        if (classifier.empty()) {
            cerr << "Error: Classifier is not trained!" << endl;
            return -1;
        }

        // 3. Подготавливаем вектор признаков для предсказания
        vector<float> all_features;
        all_features.insert(all_features.end(), query_features.shape_features.begin(),
                           query_features.shape_features.end());
        all_features.insert(all_features.end(), query_features.color_histogram.begin(),
                           query_features.color_histogram.end());
        all_features.insert(all_features.end(), query_features.texture_features.begin(),
                           query_features.texture_features.end());

        // 4. Преобразуем в формат OpenCV
        Mat sample(1, all_features.size(), CV_32F);
        for (size_t i = 0; i < all_features.size(); ++i) {
            sample.at<float>(i) = all_features[i];
        }

        // 5. Делаем предсказание
        float prediction = classifier->predict(sample);

        // 6. Возвращаем ID пуговицы
        return static_cast<int>(prediction);
    }
private:
    ButtonFeatures extractFeatures(const Mat &img) {
        ButtonFeatures features;

        // 1. Геометрические признаки (форма)
        Mat gray, binary;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 128, 255, THRESH_BINARY);

        // Находим контуры (используем копию binary, так как findContours модифицирует входное изображение)
        vector<vector<Point> > contours;
        findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // Берем наибольший контур
            auto largest_cnt = *max_element(contours.begin(), contours.end(),
                                            [](const vector<Point> &a, const vector<Point> &b) {
                                                return contourArea(a) < contourArea(b);
                                            });

            // Отношение площади к периметру
            double area = contourArea(largest_cnt);
            double perimeter = arcLength(largest_cnt, true);
            if (perimeter > 0) {
                // Защита от деления на ноль
                features.shape_features.push_back(area / perimeter);
            } else {
                features.shape_features.push_back(0);
            }

            // Отношение сторон bounding rect
            Rect rect = boundingRect(largest_cnt);
            if (rect.height > 0) {
                // Защита от деления на ноль
                features.shape_features.push_back(float(rect.width) / rect.height);
            } else {
                features.shape_features.push_back(0);
            }
        } else {
            // Если контуры не найдены, заполняем нулями
            features.shape_features = {0, 0};
        }

        // 2. Цветовые гистограммы (HSV пространство)
        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);

        int channels[] = {0, 1, 2};
        int histSize[] = {8, 8, 8};
        float h_range[] = {0, 180}; // Для H-канала в HSV диапазон 0-180
        float sv_range[] = {0, 256}; // Для S и V каналов диапазон 0-255
        const float *ranges[] = {h_range, sv_range, sv_range};

        Mat hist;
        calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);

        // Нормализуем гистограмму
        normalize(hist, hist, 1, 0, NORM_L1);

        // Преобразуем в одномерный вектор и добавляем в признаки
        hist = hist.reshape(1, 1);
        hist.convertTo(hist, CV_32F);

        features.color_histogram.assign(hist.ptr<float>(), hist.ptr<float>() + hist.cols);

        // 3. Текстура (LBP)
        Mat lbp = computeLBP(gray);

        // Параметры для гистограммы LBP
        int lbp_histSize = 256;
        float lbp_range[] = {0, 256};
        const float *lbp_ranges[] = {lbp_range};

        Mat lbp_hist;
        calcHist(&lbp, 1, 0, Mat(), lbp_hist, 1, &lbp_histSize, lbp_ranges, true, false);
        normalize(lbp_hist, lbp_hist, 1, 0, NORM_L1);

        features.texture_features.assign(lbp_hist.ptr<float>(), lbp_hist.ptr<float>() + lbp_hist.rows);

        return features;
    }

    Mat computeLBP(const Mat &gray) {
        Mat lbp = Mat::zeros(gray.size(), CV_8UC1);
        for (int y = 1; y < gray.rows - 1; ++y) {
            for (int x = 1; x < gray.cols - 1; ++x) {
                uchar center = gray.at<uchar>(y, x);
                unsigned char code = 0;
                code |= (gray.at<uchar>(y - 1, x - 1) > center) << 7;
                code |= (gray.at<uchar>(y - 1, x) > center) << 6;
                // ... остальные 6 соседей
                lbp.at<uchar>(y, x) = code;
            }
        }
        return lbp;
    }

    void trainClassifier() {
        // 1. Преобразование данных в формат OpenCV
        Mat samples(train_data.size(),
                   train_data[0].shape_features.size() +
                   train_data[0].color_histogram.size() +
                   train_data[0].texture_features.size(),
                   CV_32F);

        Mat labels(train_labels.size(), 1, CV_32S);

        // Заполнение samples и labels...

        // 2. Создание и настройка Random Forest
        Ptr<ml::RTrees> model = ml::RTrees::create();

        // Правильные параметры для OpenCV:
        model->setMaxDepth(10);  // Максимальная глубина деревьев

        // Установка критериев остановки
        TermCriteria term_crit;
        term_crit.maxCount = 100;    // Максимальное количество деревьев
        term_crit.epsilon = 0.01f;   // Требуемая точность
        term_crit.type = TermCriteria::MAX_ITER | TermCriteria::EPS;
        model->setTermCriteria(term_crit);

        // Другие доступные параметры:
        model->setCalculateVarImportance(true);  // Включить расчет важности признаков
        model->setActiveVarCount(0);             // Использовать все признаки
        model->setRegressionAccuracy(0.f);       // Для классификации

        // 3. Обучение модели
        Ptr<ml::TrainData> train_data = ml::TrainData::create(
            samples, ml::ROW_SAMPLE, labels
        );
        model->train(train_data);

        // 4. Сохранение модели
        model->save("button_classifier.yml");
        classifier = model;

        // 5. Вывод информации о важности признаков
        if (model->getCalculateVarImportance()) {
            Mat var_importance = model->getVarImportance();
            cout << "Feature importance: " << var_importance << endl;
        }
    }

    vector<ButtonFeatures> train_data;
    vector<int> train_labels;
    Ptr<ml::StatModel> classifier;
};

// Вспомогательные функции (должны быть реализованы)
float estimateSize(const Mat &binary_img, float ref_px, float ref_mm) {
    vector<vector<Point> > contours;
    findContours(binary_img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return 0.0f;

    Rect rect = boundingRect(contours[0]);
    float size_px = max(rect.width, rect.height);
    return (size_px / ref_px) * ref_mm;
}

string getButtonType(int id) {
    // Реализация получения типа по ID
    return "Type_" + to_string(id);
}

string getButtonMaterial(int id) {
    // Реализация получения материала по ID
    return (id % 2) ? "Plastic" : "Metal";
}

string getButtonDescription(int id) {
    // Реализация получения описания по ID
    return "Button description for type " + to_string(id);
}

int main() {
    // Инициализация базы данных пуговиц
    ButtonMatcher matcher;

    // 1. Загрузка и обучение на известных пуговицах
    vector<string> train_images = {
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_plast_2.jpg",
        "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug2.jpg",
    };

    cout << "Training classifier..." << endl;
    matcher.train(train_images);
    cout << "Training completed!" << endl << endl;

    // 2. Обработка пользовательского изображения
    string user_image_path = "/home/user/dir/programming/C++/Yaroslava/AmmoDetector/images/pugov/pug1_etalon.jpg";

    Mat query_image = imread(user_image_path);
    if (query_image.empty()) {
        cerr << "Error: Could not load query image!" << endl;
        return -1;
    }

    // 3. Определение типа пуговицы
    int button_id = matcher.match(query_image);
    cout << "Detected button type: " << button_id << endl;

    // 4. Определение размера (с эталонным объектом)
    float reference_size_mm = 24.0f; // Размер монеты в мм
    float reference_size_px = 150.0f; // Размер монеты в пикселях на изображении

    Mat gray;
    cvtColor(query_image, gray, COLOR_BGR2GRAY);
    float estimated_size = estimateSize(gray, reference_size_px, reference_size_mm);

    cout << "Estimated button size: " << estimated_size << "mm" << endl;

    // 5. Уточнение размера у пользователя
    float final_size;
    cout << "Please confirm button size (" << estimated_size << "mm) or enter correct value: ";
    cin >> final_size;

    // 6. Получение информации о пуговице
    cout << endl << "=== BUTTON INFORMATION ===" << endl;
    cout << "Type: " << getButtonType(button_id) << endl;
    cout << "Size: " << final_size << "mm" << endl;
    cout << "Material: " << getButtonMaterial(button_id) << endl;
    cout << "Description: " << getButtonDescription(button_id) << endl;

    // 7. Визуализация результатов
    vector<vector<Point> > contours;
    findContours(gray.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {
        drawContours(query_image, contours, -1, Scalar(0, 255, 0), 2);

        Rect rect = boundingRect(contours[0]);
        rectangle(query_image, rect, Scalar(255, 0, 0), 2);

        putText(query_image,
                format("Type: %d Size: %.1fmm", button_id, final_size),
                Point(10, 30),
                FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(0, 0, 255),
                2);
    }

    imshow("Button Analysis Result", query_image);
    waitKey(0);

    return 0;
}
