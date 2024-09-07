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

#### 문제에 대한 정보 수집
   ##### 문제 정의
Santander Customer Satisfaction에 대한 data는 고객의 만족도를 개선하기 위해 kaggle에 데이터를 제공해주었다. 따라서 Kaggle에서 고객의 정보를 토대로 Santander 은행이 제공하는 서비스에 불만족을 느끼는 고객을 식별하는 대회가 2016년에 진행되었다.

train.csv을 기반으로 적절한 EDA를 진행한 후 test.csv의 데이터를 이용해 예측한 후 결과를sample_submission.csv와 결합한 후 제출하고 제출하면 된다.

![image](https://github.com/user-attachments/assets/5026c767-5401-4087-a83c-8baaa8955859)

Santander Customer Satisfaction data는 위와 같이 모든 feature가 개인정보를 이유로 feature의 이름이 모두 익명처리 되어있다. 따라서 어떤 속성인지 추정할 수 없다.

TARGET으로는 1은 불만족, 0은 만족한 고객을 나타내며, 고객의 만족도를 예측하는 문제이다. 뿐만 아니라 평가지표로는 다음과 같이 공지되어 있다.

![image](https://github.com/user-attachments/assets/a883b253-d187-4af2-aca2-dcd435bde71e)

<www.kaggle.com/competitions/santander-customer-satisfaction/overview/evaluation>

ROC 곡선의 아래 면적을 기준으로 평가된다. 따라서 모델의 예측 성능을 AUC로 측정하며, 높은 AUC를 얻는 것이 대회에서 좋은 성적을 얻는 데 중요하다.
