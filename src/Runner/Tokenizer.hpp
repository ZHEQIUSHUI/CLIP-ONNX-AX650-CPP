#pragma once
#include "map"
#include "vector"
#include "string"
#include "fstream"

class Tokenizer
{
private:
    std::map<std::string, int> tokenizer_token2idx;

    std::vector<std::string> stringSplit(const std::string &str, char delim)
    {
        std::vector<std::string> elems;
        auto lastPos = str.find_first_not_of(delim, 0);
        auto pos = str.find_first_of(delim, lastPos);
        while (pos != std::string::npos || lastPos != std::string::npos)
        {
            elems.push_back(str.substr(lastPos, pos - lastPos));
            lastPos = str.find_first_not_of(delim, pos);
            pos = str.find_first_of(delim, lastPos);
        }
        return elems;
    }

    void tokenize(std::string token, std::vector<int> &idx)
    {
        idx.push_back(49406);
        {
            std::vector<std::string> tokens = stringSplit(token, ' ');
            for (auto t : tokens)
            {
                idx.push_back(tokenizer_token2idx[t + "</w>"]);
            }
        }
        idx.push_back(49407);

        // memset(feat, 0, sizeof(CLIP_TEXT_FEATURE_T));
        // memcpy(feat->feature, idx.data(), idx.size() * sizeof(int));
    }

    void tokenize_char(std::string token, std::vector<int> &idx)
    {
        idx.push_back(49406);
        {
            for (auto t : token)
            {
                idx.push_back(tokenizer_token2idx[std::to_string(t) + "</w>"]);
            }
        }
        idx.push_back(49407);
        // memset(feat, 0, sizeof(CLIP_TEXT_FEATURE_T));
        // memcpy(feat->feature, idx.data(), idx.size() * sizeof(int));
    }

public:
    bool load_tokenize(std::string vocab_path)
    {
        std::ifstream infile;
        infile.open(vocab_path.data());
        if (!infile.good())
        {
            return false;
        }

        std::string s;
        int idx = 0;
        while (getline(infile, s))
        {
            tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
            idx++;
        }
        infile.close();
        return true;
    }

    void encode_text(std::string text, std::vector<int> &idx)
    {
        idx.clear();
        return tokenize(text, idx);
    }

    void encode_text_char(std::string text, std::vector<int> &idx)
    {
        idx.clear();
        return tokenize_char(text, idx);
    }
};