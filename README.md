# Handwriting Calculator_using machine learning✍️


![header](https://capsule-render.vercel.app/api?type=waving&color=ADD8E6&height=300&section=header&text=Handwriting%20calculator&desc=using%20machine%20learning&fontSize=50&demo=wave&fontColor=696969)

# :pushpin:Project Description

Teachable machine을 이용해 사용자가 필기체로 입력한 수식을 인식하여 계산해주는 계산기 프로그램이다.
--------------------------------------------------------
# :pushpin:Project video

https://user-images.githubusercontent.com/105347300/205647125-74062a5f-0be9-419a-8472-40762602ea31.mp4

youtube : https://www.youtube.com/watch?v=JaJqFwNpuyE﻿

---------------------------------------------------------
# :pushpin:Project function

- GUI : 사용자와 상호작용하는 계산기 화면 기능  
- 문자인식 : 18개의 문자를 인식하는 기능  
 0,1,2,3,4,5,6,7,8,9,(,A,N,S,+,-,x,/  
- 계산기 기능 : 계산(=) , ANS , CE , AC기능  
![image](https://user-images.githubusercontent.com/105347300/205648138-54bd5977-ac56-44d4-9db3-47a5f13a4ada.png)

# :pushpin:Project Block diagram

![image](https://user-images.githubusercontent.com/105347300/205648749-8a029d2f-93d6-4fdd-b4c7-5b9a26d010ba.png)
----------------------------------------------------------

# :pushpin:Project algorithm
## 문자학습 
- Teachable Machine -google 사용
---------------------------
## 모델 파일 변환
- google colab 에서 상기의 convert.ipynb파일 실행 후 openCV용 모델 파일로 변환 
----------------------------
## 필기체 입력 기능 
- 사용자의 마우스 입력의 마지막 좌표를 저장해 현재 좌표와 라인을 이어서 실시간으로 그리는 함수
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
## =(등호)버튼 (객체 레이블링 기능)
- 사용자의 필기체 문자열입력 구간을 레이블링해 객체를 찾아내고, 객체의 좌표를 벡터에 저장하는 함수
- ![image](https://user-images.githubusercontent.com/105347300/205651154-f45aefa9-1030-4f10-b05e-34bd2a964165.png)
-     void Object_Recognition(Mat img, vector<Rect>& r) {   //필기체 입력구간 객체 인식, 레이블링 하는 함수
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
--------------------------
 ## 문자 인식 기능
 - opencv용 모델파일을 불러와 블롭객체에 넣고, 이 블롭 객체를 그대로 네트워크 입력으로 설정하고, 순방향으로 실행해 예측 결과 행렬을 얻어 문자 추론  
 - ![image](https://user-images.githubusercontent.com/105347300/205651819-8fdf0d52-da90-49e6-a7d7-fdd95da8e5d0.png)

-     void tm_machine(Mat dst, vector<Rect>& r) //문자인식 기능 함수
    {
        for (int i = 1; i < r.size(); i++) //객체 인식
        {
            if (r[i].width < 100 && r[i].height < 100) { //객체 높이와 넓이가 100이하이면 소수점 처리
                message += "."; // 예측결과 넣기
                cout << "예측 결과 : 소수점 ." << endl;
            }
            else { //소수점이 아니면 문자 인식 수행
                Mat img =dst(r[i]); //레이블링된 객체 이미지 추출

                vector<String> classNames = { "0","1","2","3","4","5","6","7","8","9","/","+","-","x","(","A","N","S" }; //클래스 이름 
                // Load network
                Net net = readNet("frozen_model.pb"); //모델 파일 불러오기
                if (net.empty()) { cerr << "Network load failed!" << endl; } //에러처리
                // Load an image
                if (img.empty()) { //에러처리
                    cerr << "Image load failed!" << endl;
                }
                // Inference
                Mat predict; //예측 이미지
                cvtColor(img, predict, COLOR_GRAY2RGB); //채널변경

                Mat inputBlob = blobFromImage(predict, 1 / 127.5, Size(224, 224), -1, 0); //블롭 객체 생성
                net.setInput(inputBlob);//네트워크 입력으로 설정
                Mat prob = net.forward(); //네트워크를 실행
                // Check results & Display

                double maxVal; // 최대값 저장할 변수
                Point maxLoc; //최대값 위치 저장할 변수
                minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
                if (classNames[maxLoc.x] == "A" || classNames[maxLoc.x] == "N" || classNames[maxLoc.x] == "S") { //문자처리
                    if (classNames[maxLoc.x] == "S") { //s가 인식되면 
                        message += to_string(ans); //ans값 대입
                    }
                }
                else if (classNames[maxLoc.x] == "(") { //괄호 처리
                    if (cnt == 0) { // 여는 괄호
                        message += classNames[maxLoc.x]; // ( 넣기!!!!!!!!!
                        cnt = 1;
                    }
                    else if (cnt == 1) {//닫는 괄호이면
                        message += ")"; // ( 넣기!!!!!!!!!
                        cnt = 0;
                    }
                }
                else {
                    message += classNames[maxLoc.x]; // 예측결과 넣기!!!!!!!!!
                }
                String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal *
                    100);
                cout << "예측 결과" << str << endl;//예측 결과 출력
                
                img = Scalar(0, 0, 0); //창 지우기

            }

        }

    }
-----------------------------
## 계산 기능
:paperclip:계산기 구현 방법 
-----------------

### 식을 후위표기법으로 바꾸기 --- > string calc1();


  >1. 숫자가 나오면 그대로 문자열에 입력하고 ' '로 구분한다.

  >2. '(' 나오면 연산자 벡터에 push한다.

  >3. '*' '/' 나오면 연산자 벡터에 push한다.

  >4. '+' '-' 연산이 나오면 연산자 벡터에있는 여는 괄호 이후부터 끝까지 문자열 변수에 입력하고 , 여는 괄호가 없다면 연산자 벡터의 끝까지 문자열 변수에 넣고 이 연산자를 연산자 벡터에 push한다.

  >5. 닫는 괄호(')')가 나오면 여는 괄호('(')가 나올때까지 문자열 변수에 입력한다. 괄호는 넣지 않는다.


### 후위표기법 계산하기 --> double calc2(string str);


  >1.문자열 앞에서부터 ' ' 공백을 기준으로 끊기

  >2. 숫자이면 벡터에 push

  >3. 연산자이면 벡터의 마지막 요소와 마지막 전 요소를 연산후 두개의 요소를 pop하고 결과값을 다시 벡터에 push

----------------------------------------------

- 문자열을 후위 표기법으로 만들기 기능 함수
 - ![image](https://user-images.githubusercontent.com/105347300/205653749-f4c4f497-2e65-4b65-a800-671565eb94bc.png)
 
     string calc1() {  //문자열을 후위 표기법으로 만들기 기능 함수
        vector<char> op;// 연산자를 저장할 벡터
        string num = ""; // 숫자를 저장할 문자열 변수
        //문자를 숫자로
        int start = 0; 
        int ox = 0;
        int i, num_cnt = 0;
        for (i = 0; i < message.size(); i++) // 후위 표기법 만들기
        {
            if (((int)message[i] <= 57 && (int)message[i] > 47) && i == message.size() - 1) { // 문자열의 맨마지막이고  문자가 숫자이면
                num_cnt++;
                string s = message.substr(start, num_cnt);// 숫자
                num += s; //연산자 앞 숫자 자른 문자열 숫자로 바꿔서 넣기
                num += " ";
            }
            else if (((int)message[i] <= 57 && (int)message[i] > 47) || message[i] == '.') { //문자가 숫자 이거나 소수점이면

                num_cnt++; // 숫자 자리수,카운트하기
            }
            else { // 연산자이면
                if ((int)message[i - 1] <= 57 && (int)message[i - 1] > 47) { //연산자 앞이 숫자이면
                    string s = message.substr(start, num_cnt);//연산자 앞 숫자 만 자르기
                    num += s; //연산자 앞 숫자를 자른 문자열 숫자로 바꿔서 변수에 저장
                    num += " "; // 한 숫자 구분 띄어쓰기.
                    num_cnt = 0; //숫자 자리수 0개로 대입.
                }
                start = i + 1; //연산자 뒷 숫자의 인덱스
                if (op.empty()) {//연산자가 없으면, 첫 연산자가 등장시
                    op.push_back((char)message[i]); //연산자 넣기
                }
                else if (message[i] == '(' || message[i] == 'x' || message[i] == '/') { //이 연산자이면 벡터에 push
                    op.push_back((char)message[i]);
                }
                else if (message[i] == '+' || message[i] == '-') { //여는 괄호있으면  (위의 모든 연산자 출력, (없으면 벡터의 끝까지 출력 후 스택에push
                    for (int j = 0; j < op.size(); j++) //'+' '-' 연산이 나오면 여는 괄호 이후부터 끝까지 문자열에 입력
                    {
                        if (op[j] == '(') { //괄호 안  연산자 출력
                            if (op.size() - 1 == j) { // 괄호 위에 아무것도 없으면
                            }
                            else {
                                //여는 괄호 이후부터 끝까지 문자열에 입력
                                for (int k = op.size() - 1; k >= j + 1; k--)
                                {
                                    num += op[k];
                                    num += " "; // 구분

                                    op.pop_back(); // 연산자 벡터에서 마지막 연산자 삭제
                                }
                            }
                            ox++;
                        }
                    }
                    if (ox == 0) // 괄호 없으면 스택의 끝까지 출력
                    {
                        for (int k = op.size() - 1; k >= 0; k--)
                        {
                            num += op[k]; // 출력
                            num += " "; // 구분
                            op.pop_back();// 연산자 벡터에서 마지막 연산자 삭제

                        }
                    }
                    op.push_back((char)message[i]); // push
                    ox = 0;
                }
                else if (message[i] == ')') { // 닫는 괄호 나오면 여는 괄호까지 모든 op비우기

                    for (int j = 0; j < op.size(); j++)
                    {
                        if (op[j] == '(') {//여는 괄호까지 모든 op비우기
                            for (int k = op.size() - 1; k >= j + 1; k--)
                            {
                                num += op[k];//출력
                                num += " "; // 구분
                                op.pop_back();// 연산자 벡터에서 마지막 연산자 삭제

                            }

                            op.pop_back();// 연산자 벡터에서 마지막 연산자 삭제
                        }
                    }
                }
            }
        }
        if (i == message.size()) { // message의 식이 끝나면 벡터에 남아있는 연산자 모두 num에 출력
            for (int j = op.size() - 1; j >= 0; j--)
            {

                num += op[j]; //남아있는 연산자 넣기
                num += " "; // 구분
            }
        }
        cout << num << endl; // 후위표기법 출력
        return num;// 후위표기법 출력
    }
 -----------------------------------------------------------
- 후위표기법을 계산하는 함수
- ![image](https://user-images.githubusercontent.com/105347300/205653770-695d5fd3-6d6e-471e-99bf-9a12bb0b0734.png)
-     double calc2(string str) { // 후위표기법 계산
        vector<double> v;
        int cur_position = 0; // 시작위치
        int position; // 공백 인덱스
        int cnt = 0;
        double val1, val2, value = 0;
        bool ox = false;
        while ((position = str.find(" ", cur_position)) != std::string::npos) {//문자열 앞에서부터 ' ' 공백을 기준으로 끊기
            int len = position - cur_position;//공백을 기준으로 끊은 문자의 개수
            string result = str.substr(cur_position, len);//문자열 시작위치에서 len개수만큼 잘라낸 문자열
            for (int i = 0; i < result.size(); i++) {
                if (((int)result[i] <= 57 && (int)result[i] > 47) || result[i] == '.') { // 숫자이거나 소수점이면
                    ox = true; // 숫자이면 벡터에 push
                    cnt++; //숫자이면 
                }
            }
            if (ox) { // 숫자이면
                v.push_back(stod(result));//숫자이면 벡터에 push
                ox = false;
            }
            if (cnt == 0) { //연산자이면 벡터의 마지막 요소와 마지막 전 요소끼리 연산후 두개의 요소를 pop하고 결과값을 벡터에 push

                val2 = v.back();// 벡터의 마지막 요소
                val1 = v[v.size() - 2];//마지막 전 요소
                //cout << "val1:" << val1 << "val2:" << val2 << endl;
                calc_op(&val1, &val2, cur_position, str);//벡터의 마지막 요소와 마지막 전 요소끼리 연산
                v.pop_back();//두개의 요소를 pop
                v[v.size() - 1] = val1; // 결과값을 벡터에 push
            }
            cur_position = position + 1; //시작위치 갱신
            cnt = 0;
        }
        value = v[0]; //계산 결괏값
        cout << "답:" << v[0] << endl;////계산 결괏값 출력
        return value;
    }
 ---------------------------------------------------------
- 문자 연산자에따른 두개의 값을 연산하는 함수
- ![image](https://user-images.githubusercontent.com/105347300/205653707-e1d2d47b-1852-40a6-829e-a562af4124ec.png)
 -     void calc_op(double* val1, double* val2, int op_index, string str) { //두개의 값 연산하는 함수
        if ((int)str[op_index] == 120) { //곱셈
            *val1 = (*val1) * (*val2);
        }
        else if ((int)str[op_index] == 45) { //-
            *val1 -= (*val2);
        }
        else if ((int)str[op_index] == 47) {// /
            *val1 /= (*val2);
        }
        else if ((int)str[op_index] == 43) { //덧셈
            *val1 += *val2;
        }
    }
 -----------------------------------------------
## ANS기능
- 클릭시 계산한 수식의 값을 저장하고 필기체 입력칸에 ANS가 인식되었을시 저장되었던 결괏값을 ANS에 대입.

- ![image](https://user-images.githubusercontent.com/105347300/205661595-188fb756-7a27-4e00-aed7-9dab28362619.png)
-             else if (x >= 1300 && (y < 300 && y>200)) { // ans구간 설정
                if (level == 1) { // =버튼 입력 뒤에 사용가능
                    ans = value; //값 대입
                    cout << "ans:" << ans << endl; //출력
                    draw_q(img, dst, ans, 0); //값 출력
                    draw(img);
                    imshow("img", img);

                }
            }
 
 --------------------
 ## AC기능
 -입력기능 창에 Scalar(0)을 대입해 사용자의 전체 입력 삭제
- ![image](https://user-images.githubusercontent.com/105347300/205662175-f639c0ed-79e3-4d7d-9ae4-7c30435870ef.png)
- else if (x >= 1300 && y <= 100) { // ac버튼 구간 설정
       img = Scalar(0); // 필기체 입력칸 지우기
       draw(img); // 필기체 구성 그리기 함수 호출

       level = 2;
   }

                                            
--------------------------
## CE기능
- 필기체 입력 구간을 레이블링해서 사용자의 마지막 입력을 알아내 삭제하는 기능 
- ![image](https://user-images.githubusercontent.com/105347300/205662391-5222cb15-5d69-44ee-834c-adb450d03fa5.png)
-     void ce_button(Mat img) { //ce버튼 구현 함수
        Mat labels, stats, centroids;
        int cnt = connectedComponentsWithStats(img, labels, stats, centroids);//객체 레이블링
        //cout << stats.rows<<"개의 레이블 배경포함 객체수"<<endl;
        vector<Rect> r;
        for (int i = 0; i < cnt; i++)      //바운딩 박스 정보 vector<Rect>r에 저장
        {
            int* p = stats.ptr<int>(i); //행 역역 추출
            r.push_back(Rect(p[0], p[1], p[2], p[3]));  //바운딩 박스 정보 vector<Rect>r에 저장
        }
        Rect tmp;
        for (int i = 1; i < r.size(); i++)      // 벡터 r의 저장된 순서를 x좌표가 작은순부터 저장되도록 정렬
        {
            for (int j = 0; j < r.size() - i; j++)
            {
                if (r[j].x > r[j + 1].x) {// x좌표가 작은순부터 저장되도록 정렬
                    tmp = r[j];
                    r[j] = r[j + 1];
                    r[j + 1] = tmp;
                }
            }
        }
        int x = r[r.size() - 1].x; //마지막에 그려진 객체 영역 설정
        int y = r[r.size() - 1].y; //마지막에 그려진 객체 영역 설정
        int w = r[r.size() - 1].width; //마지막에 그려진 객체 영역 설정
        int h = r[r.size() - 1].height; //마지막에 그려진 객체 영역 설정
        Mat last = img(Rect(x, y, w, h)); //마지막에 그려진 객체 영역 설정
        last = Scalar(0); // 객체 영역 초기화
    }
