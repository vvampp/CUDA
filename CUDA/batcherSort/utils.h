#ifndef UTILS_H
#define UTILS_H

#include <string>

float* txt_to_array(const std::string& file_name, long& size);
void array_to_txt(const std::string& file_name, const float* array, long size);

#endif
