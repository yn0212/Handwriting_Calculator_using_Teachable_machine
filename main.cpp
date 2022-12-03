#include "yn.h"
using namespace jyn;
int main(void) {
	Mat img(500, 1500, CV_8UC1, Scalar(0)); //gui화면 출력
	draw(img); //gui 화면 구성 그리기 함수

	setMouseCallback("img", onMouse, &img); //마우스 콜백함수 출력
	imshow("img", img); //영상 출력
	waitKey(); //무한 대기

	return 0; //0을 외부로 반환
}
