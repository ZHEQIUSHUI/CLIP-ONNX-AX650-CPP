#include "internal_func.hpp"

#include "math.h"
#include "string"
#include "fstream"
#include "vector"

void _normalize(float *feature, int feature_len)
{
    float sum = 0.0f;
    for (size_t i = 0; i < feature_len; i++)
    {
        sum += feature[i] * feature[i];
    }
    sum = 1.0f / sqrt(sum);
    for (int it = 0; it < feature_len; it++)
        feature[it] *= sum;
}

float _compare(float *feature1, float *feature2, int feature_len)
{
    float similarity = 0.0;
    for (int i = 0; i < feature_len; i++)
        similarity += feature1[i] * feature2[i];
    similarity = similarity < 0 ? 0 : similarity > 1 ? 1
                                                     : similarity;
    return similarity;
}

bool _file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

bool _file_read(const std::string &path, std::vector<char> &data)
{
    std::fstream fs(path, std::ios::in | std::ios::binary);

    if (!fs.is_open())
    {
        return false;
    }

    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg = fs.tellg();

    auto file_size = static_cast<size_t>(fs_end - fs_beg);
    auto vector_size = data.size();

    data.reserve(vector_size + file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();

    return true;
}

bool _file_dump(const std::string &path, char *data, int size)
{
    std::fstream fs(path, std::ios::out | std::ios::binary);

    if (!fs.is_open() || fs.fail())
    {
        fprintf(stderr, "[ERR] cannot open file %s \n", path.c_str());
    }

    fs.write(data, size);

    return true;
}