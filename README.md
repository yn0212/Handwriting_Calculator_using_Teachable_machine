# Handwriting Calculator_using machine learning✍️


![header](https://capsule-render.vercel.app/api?type=waving&color=ADD8E6&height=300&section=header&text=Handwriting%20calculator&desc=using%20machine%20learning&fontSize=50&demo=wave&fontColor=696969)

## :pushpin:Project Description

Teachable machine을 이용해 사용자가 필기체로 입력한 수식을 인식하여 계산해주는 계산기 프로그램이다.
--------------------------------------------------------
## :pushpin:Project video

https://user-images.githubusercontent.com/105347300/205647125-74062a5f-0be9-419a-8472-40762602ea31.mp4

youtube : https://www.youtube.com/watch?v=JaJqFwNpuyE﻿

---------------------------------------------------------
## :pushpin:Project function

- GUI : 사용자와 상호작용하는 계산기 화면 기능  
- 문자인식 : 18개의 문자를 인식하는 기능  
 0,1,2,3,4,5,6,7,8,9,(,A,N,S,+,-,x,/  
- 계산기 기능 : 계산(=) , ANS , CE , AC기능  
![image](https://user-images.githubusercontent.com/105347300/205648138-54bd5977-ac56-44d4-9db3-47a5f13a4ada.png)

## :pushpin:Project Block diagram

![image](https://user-images.githubusercontent.com/105347300/205648749-8a029d2f-93d6-4fdd-b4c7-5b9a26d010ba.png)
----------------------------------------------------------

## :pushpin:Project algorithm
### 문자학습 
- Teachable Machine -google 사용
---------------------------
### 모델 파일 변환
- google colab 에서 상기의 convert.ipynb파일 실행 후 openCV용 모델 파일로 변환 
----------------------------
### 필기체 입력 기능 
- ![image](https://user-images.githubusercontent.com/105347300/205650161-ad97a274-67c8-42c1-aea4-bce1ae2670c5.png)
-         else if (event == EVENT_MOUSEMOVE) { //필기체 입력 그리기
            if (flags & EVENT_FLAG_LBUTTON) {
                if (x < 1300 && y >= 100) { //구간 설정
                    line(img, ptOld, Point(x, y), Scalar(255), 10); //그리기
                    imshow("img", img); //영상 출력
                    ptOld = Point(x, y); //마지막 좌표 저장

                }
            }
        }
---------------------------
### =(등호)버튼 (객체 레이블링 기능)
- ![image](https://user-images.githubusercontent.com/105347300/205651154-f45aefa9-1030-4f10-b05e-34bd2a964165.png)
- void Object_Recognition(Mat img, vector<Rect>& r) {   //필기체 입력구간 객체 인식, 레이블링 하는 함수
        Mat labels, stats, centroids; //객체 레이블링에 필요한 변수 생성
        int cnt = connectedComponentsWithStats(img, labels, stats, centroids);// 객체 레이블링

        for (int i = 0; i < cnt; i++)      //바운딩 박스 정보 vector<Rect>r에 저장
        {
            int* p = stats.ptr<int>(i); //i행 단위 정보 추출
            r.push_back(Rect(p[0]-30, p[1]-30, p[2]+60, p[3]+60)); // 바운딩 박스 정보에 여백을 포함한 크기를 벡터에 저장
        }

        Rect tmp; // 레이블링된 객체를 벡터r에 저장된 순서를 바꾸기위한 빈 객체 생성
        for (int i = 1; i < r.size(); i++)      // 벡터 r의 저장된 순서를 x좌표가 작은순부터 저장되도록 정렬
        {
            for (int j = 0; j < r.size() - i; j++) 
            {       // x좌표가 더 작은 객체가 작은 인덱스를 가지도록 설정
                if (r[j].x > r[j + 1].x) {// 객체의 x좌표 비교
                    tmp = r[j]; //x좌표가 큰 객체를대입
                    r[j] = r[j + 1]; //x좌표가 더 작은 객체 대입해 위치 변경
                    r[j + 1] = tmp; //대입
                }
            }
        }

    }
 
