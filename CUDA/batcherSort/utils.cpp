#include "utils.h"
#include <fstream>
#include <iostream>

float* txt_to_array(const std::string& file_name, long& size){
        std::ifstream input_file(file_name);
        if(!input_file.is_open()) {
                std::cerr << "Error while opening the input file." << std::endl;
                size = 0;
                return nullptr;
        }

        size = 0;
        float temp_num;
        while(input_file >> temp_num){
                size ++;
        }

        input_file.clear();
        input_file.seekg(0, std::ios::beg);

        if(size == 0)
                return nullptr;

        float* array = new float[size];

        for( int i = 0 ; i < size ; ++i )
                input_file >> array[i];

        return array;
}

void array_to_txt(const std::string& file_name, const float* array, long size){
        std::ofstream output_file(file_name);
        if(!output_file.is_open()) {
                std::cerr << "Error while creating the output file." << std::endl;
                return;
        }

        for( int i = 0 ; i < size ; ++ i)
                output_file << array[i] << "\n";

        std::cout << "Data successfully written on '" << file_name << "'." << std::endl;
        output_file.close();
}
