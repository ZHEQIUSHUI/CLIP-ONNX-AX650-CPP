#pragma once
#include "string"
#include "vector"

void _normalize(float *feature, int feature_len);
float _compare(float *feature1, float *feature2, int feature_len);


bool _file_exist(const std::string &path);
bool _file_read(const std::string &path, std::vector<char> &data);
bool _file_dump(const std::string &path, char *data, int size);