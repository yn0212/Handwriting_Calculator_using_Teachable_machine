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

### 모델 파일 변환
- google colab 에서 상기의 convert.ipynb파일 실행 후 openCV용 모델 파일로 변환 

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

