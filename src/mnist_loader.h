#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

// Loads at most maxRows rows from an MNIST CSV file.
// Each row: label, pixel_0, pixel_1, ..., pixel_783
// Returns pixel values normalized to [0,1], and one-hot labels.
inline void loadMNIST(const std::string& path, int maxRows,
                      std::vector<float>& images,   // maxRows * 784
                      std::vector<float>& labels) { // maxRows * 10, one-hot
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        exit(1);
    }

    images.clear();
    labels.clear();

    std::string line;
    int rowCount = 0;
    while (std::getline(file, line) && rowCount < maxRows) {
        std::stringstream ss(line);
        std::string cell;

        std::getline(ss, cell, ',');
        int label = std::stoi(cell);

        // one-hot encode: 10 floats, all 0 except a 1 at index `label`
        for (int c = 0; c < 10; c++) {
            labels.push_back(c == label ? 1.0f : 0.0f);
        }

        // remaining 784 values are pixel intensities 0-255; normalize to 0-1
        while (std::getline(ss, cell, ',')) {
            images.push_back(std::stof(cell) / 255.0f);
        }

        rowCount++;
    }

    std::cout << "Loaded " << rowCount << " rows from " << path << std::endl;
}
