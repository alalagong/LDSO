#include "frontend/SemanticRW.h"

namespace ldso {

    namespace IOWrap {

        inline bool is_base64(const char c)
        {
            return (isalnum(c) || (c == '+') || (c == '/'));
        }

        std::string base64_decode(std::string const &encoded_string)
        {
            int in_len = (int)encoded_string.size();
            int i = 0;
            int j = 0;
            int in_ = 0;
            unsigned char char_array_4[4], char_array_3[3];
            std::string ret;

            while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
            {
                char_array_4[i++] = encoded_string[in_];
                in_++;
                if (i == 4)
                {
                    for (i = 0; i < 4; i++)
                        char_array_4[i] = base64_chars.find(char_array_4[i]);

                    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                    for (i = 0; (i < 3); i++)
                        ret += char_array_3[i];
                    i = 0;
                }
            }
            if (i)
            {
                for (j = i; j < 4; j++)
                    char_array_4[j] = 0;

                for (j = 0; j < 4; j++)
                    char_array_4[j] = base64_chars.find(char_array_4[j]);

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (j = 0; (j < i - 1); j++)
                    ret += char_array_3[j];
            }

            return ret;
        }

        cv::Mat base2Mat(std::string &base64_data)
        {
            cv::Mat img;
            std::string s_mat;
            s_mat = base64_decode(base64_data);
            std::vector<char> base64_img(s_mat.begin(), s_mat.end());
            img = cv::imdecode(base64_img, CV_LOAD_IMAGE_COLOR);
            return img;
        }

        MinimalImageB *readSemanticLabel_8U(std::string filename) 
        {
            cv::Mat label_img;

            YAML::Node config = YAML::LoadFile( filename.c_str() );
            YAML::Node instance = config["instance"];       // instance

            for(int i=0; i<instance.size(); ++i)
            {
                int label_id = instance[i]["category_id"].as<int>();

                // all labels img
                if( label_id == -1)
                {
                    YAML::Node box = instance[i]["box"];
                    if(!(box[0].as<int>()==0 && box[1].as<int>()==0))
                    {
                        std::cout<<"size of labels image error in yaml"<<std::endl;
                        break;
                    }
                    std::string label_string = instance[i]["segmentation"].as<std::string>();
                    label_img = base2Mat(label_string);
                    break;
                }
            }

            cv::cvtColor(label_img, label_img, CV_RGB2GRAY);
            if(label_img.type() != CV_8U )
            {
                printf("label_img read something wrong! this may segfault. \n");
                return 0;
            }

            MinimalImageB *img = new MinimalImageB(label_img.cols, label_img.rows);
            memcpy(img->data, label_img.data, label_img.rows * label_img.cols);
            return img;

        }


        MinimalImageB *readSemanticBel_8U(std::string filename) 
        {
            cv::Mat confidence_img;

            YAML::Node config = YAML::LoadFile( filename.c_str() );
            std::string confid_string = config["confidence"].as<std::string>();  // confidence 

            confidence_img = base2Mat(confid_string);

            cv::cvtColor(confidence_img, confidence_img, CV_RGB2GRAY);
            if(confidence_img.type() != CV_8U)
            {
                printf("confidence_img read something wrong! this may segfault. \n");
                return 0;
            }

            MinimalImageB *img = new MinimalImageB(confidence_img.cols, confidence_img.rows);
            memcpy(img->data, confidence_img.data, confidence_img.rows * confidence_img.cols);
            return img;
    }

} // end IOWrap 
} // end ldso
