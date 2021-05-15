#pragma once
#ifndef LDSO_SEMANTIC_RW_H_
#define LDSO_SEMANTIC_RW_H_

#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "NumTypes.h"
#include "MinimalImage.h"

namespace ldso {

namespace IOWrap {
    const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    inline bool is_base64(const char c);
    std::string base64_decode(std::string const & encoded_string);
    cv::Mat base2Mat(std::string &base64_data);

    MinimalImageB *readSemanticLabel_8U(std::string filename);
    
    MinimalImageB *readSemanticBel_8U(std::string filename);

}

} 



#endif // LDSO_SEMANTIC_RW_H_

