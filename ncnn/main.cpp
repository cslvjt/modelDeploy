#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "net.h"
using namespace std;
int main(){
    string img_path="face.png";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    int input_width=256;
    int input_height=256;

    ncnn::Net net;
    net.load_param("srcnn_sim-opt.param");
    net.load_model("srcnn_sim-opt.bin");

    ncnn::Mat input=ncnn::Mat::from_pixels(img.data,ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    ncnn::Mat output;
    ncnn::Extractor extractor=net.create_extractor();
    extractor.input("input",input);
    extractor.extract("output",output);
    cv::Mat result(output.h,output.w,CV_8UC3);
    output.to_pixels(result.data,ncnn::Mat::PIXEL_BGR);
    cv::imwrite("srcnn.png",result);
    cout<<"Finish"<<endl;
    return 0;
}