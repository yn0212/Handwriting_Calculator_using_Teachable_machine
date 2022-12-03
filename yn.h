
#ifndef YN_H
#define YN_H
#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>
#include <iostream>
#include<string>
#include<stdlib.h>
using namespace std;
using namespace cv::dnn;
using namespace cv;
namespace jyn
{


    extern int cnt ; // 문자열의 괄호 개수를 세는 전역 변수
    extern String message; //인식된 문자열을 저장하는 전역 변수
    extern double ans;
    void calc_op(double* val1, double* val2, int op_index, string str);   //두개의 값 연산하는 함수
    string calc1();                                                      //문자열을 후위 표기법으로 만들기 기능 함수
    double calc2(string str);                                           // 후위표기법 계산하는 함수
    void ce_button(Mat img);                                           //ce버튼 구현 함수
    void Object_Recognition(Mat img, vector<Rect>& r);                // =버튼 구현 함수, 객체 인식& 객체 레이블링
    void draw(Mat img);                                                    //gui 화면 구성 그리기 함수
    void onMouse(int event, int x, int y, int flags, void* userdata);     //마우스 콜백 함수
    void draw_q(Mat img, Mat dst, double value, int ox);                // 계산 결과 출력 기능 함수
    void tm_machine(Mat dst, vector<Rect>& r);

}
#endif