### 👨‍🏫 Santander Customer Satisfaction - Machine Learning from Disaster
kaggle에서 제공하는 Santander Customer Satisfaction를 이용해 EDA와 model 학습을 통해 고객 만족도를 예측하는 프로젝트

---
### ⏲️ 분석 기간
2024.09.01 - 2024.09.03

---

### 📝 소개
Santander Customer Satisfaction data는 이전에 진행했던 titanic data와 다르게 feature의 갯수도 많으며 데이터의 양도 많다. 뿐만 아니라 feature의 대부분이 개인정보를 이유로 feature의 이름이 공개되지 않은 데이터입니다.

따라서 이번엔 분석할 Santander Customer Satisfaction은 titanic data와 다른 의미로 어려움이 있을 수 있습니다.

---

### 프로젝트 개요
##### 📌 목표
Kaggle에서 고객의 정보를 토대로 Santander 은행이 제공하는 서비스에 불만족을 느끼는 고객을 식별하는 대회가 2016년에 진행되었습니다. 
이 프로젝트의 목표는 고객이 Santander 은행 서비스에 만족하는지 불만족하는지를 예측하는 것입니다. 데이터셋은 feature의 이름이 모두 익명처리 되어있습니다. 이러한 특징들을 분석함으로써, 고객 만족도를 정확하게 예측하는 모델을 개발하고자 합니다.

##### 🖥️ 데이터셋 (Data Set)
이 프로젝트에서 사용한 데이터셋은 Kaggle에서 제공하는 다음 파일들로 구성되어 있습니다.
1. train.csv: 훈련 데이터셋, 특징들과 목표 변수를 포함.
2. test.csv: 테스트 데이터셋, 예측을 위해 사용될 데이터.
3. sample_submission.csv: 예측 결과를 제출하기 위한 샘플 파일.

---

##### 방법론
1. 문제에 대한 정보 수집
  * 문제 정의
  * 분석 대상에 대한 이해
2. Santander Customer Satisfaction data set을 이용한 EDA
  * 공통 코드
    * 오차행렬(Confusion matrix) 및 평가 지표
  * 분석
    * Santander Customer Satisfaction data set에 대한 기본적인 정보
    * feature 분석
    * 이상치 탐색
    * Data cleaning
    * Feature Engineering
3. 모델 학습
  * XGBoost
  * LightGBM
  * CatBoost
  * Ensemble
4. 결론
  * 한계점

---
