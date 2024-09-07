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
   ##### 1. 문제 정의
Santander Customer Satisfaction에 대한 data는 고객의 만족도를 개선하기 위해 kaggle에 데이터를 제공해주었다. 따라서 Kaggle에서 고객의 정보를 토대로 Santander 은행이 제공하는 서비스에 불만족을 느끼는 고객을 식별하는 대회가 2016년에 진행되었다.

train.csv을 기반으로 적절한 EDA를 진행한 후 test.csv의 데이터를 이용해 예측한 후 결과를sample_submission.csv와 결합한 후 제출하고 제출하면 된다.

![image](https://github.com/user-attachments/assets/5026c767-5401-4087-a83c-8baaa8955859)

Santander Customer Satisfaction data는 위와 같이 모든 feature가 개인정보를 이유로 feature의 이름이 모두 익명처리 되어있다. 따라서 어떤 속성인지 추정할 수 없다.

TARGET으로는 1은 불만족, 0은 만족한 고객을 나타내며, 고객의 만족도를 예측하는 문제이다. 뿐만 아니라 평가지표로는 다음과 같이 공지되어 있다.

![image](https://github.com/user-attachments/assets/a883b253-d187-4af2-aca2-dcd435bde71e)

<www.kaggle.com/competitions/santander-customer-satisfaction/overview/evaluation>

ROC 곡선의 아래 면적을 기준으로 평가된다. 따라서 모델의 예측 성능을 AUC로 측정하며, 높은 AUC를 얻는 것이 대회에서 좋은 성적을 얻는 데 중요하다.

---

   ##### 2. 분석 대상에 대한 이해
Santander 은행은 스페인 산탄데르에 1867년에 설립되 유럽 최대 기업 및 은행이다. Santander 은행은 다른 세계적인 은행과 다른 특징이 있다. 대형 은행들은 투자금융 분야 규모가 크다. 하지만 Santander 은행은 수익의 큰 부분이 소매금융에서 나온다. 즉, 금융기관인 Santander 은행이 개인에게 금융 서비스를 제공하는 것이 수익의 큰 부분이다.

소매금융에서 나오는 수익이 크기 때문에 Santander 은행은 고객의 만족, 불만족에 큰 관심을 가지게 된 것으로 kaggle에 feature 이름이 익명처리된 data를 제공한 것이다.

고객을 대상으로 하는 많은 분야에서 고객 만족도는 성공의 중요한 척도이다. 기업에 불만족을 느끼는 고객은 더이상 고객으로 머물지 않는다. 하지만 대부분의 고객은 기업에서 제공하는 서비스에 대한 불만족을 떠나기 전까지 혹은 떠난 후에도 거의 표출하지 않는다.

따라서 Santander Bank는 kaggle에 고객에 대한 데이터를 제공하면서 초기 단계에서 불만족한 고객을 식별하는 데 도움을 요청했다. 즉, Santander Bank는 고객이 떠나기 전에 고객의 만족을 개선하기 위한 선제적인 조치를 취하고자 한 것이다.

Santander Customer Satisfaction data는 앞에서 말했듯 수백 개의 익명화된 특징을 사용하여 고객이 은행 경험에 만족하는지 불만족하는지를 예측하고자 한다. 필자는 이러한 특징으로 인해 Santander Customer Satisfaction 데이터를 마냥 쉬운 난이도로 보고있지 않다.

---

#### Santander Customer Satisfaction data set을 이용한 EDA
   ##### 1. 공통 코드
```
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, pos_label=1)
    recall = recall_score(y_test, pred, pos_label=1)
    f1 = f1_score(y_test, pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```
분류 모델의 성능을 평가하기 위해 공통 코드로 위와 같이 오차행렬, 정확도, 재현율, 정밀도, F1 score, ROC 곡선과 AUC를 평가하기 위해 다음과 같이 코드를 작성했다. 공통 코드는 모델을 사용했을 때마다 성능을 평가하기 위해 사용할 것으로 함수화 했다. 각각 설명은 다음과 같다.

1. 오차행렬
혼동행렬(confusion matrix)라고도 부른다. 학습된 분류 모델이 예측을 수행하면서 얼마나 혼동(혼란)하고 있는지 보여주는 지표이다. 따라서 이진 분류에서 예측 오류가 얼마인지, 어떤 유형의 예측 오류가 발생하고 있는지 보여주는 지표이다.
![image](https://github.com/user-attachments/assets/aa79ad40-6ba9-4129-9545-ecb01fe83cb0)

위와 같이 총 4개 분면을 통해 분류 모델 예측 성능을 보여주는 지표이다. TN, FP, FN, TP가 있으며 각각 다음과 같은 지표를 보여준다.

   ###### TN = 예측 값을 Negative 값으로 예측했고 실제 값이 Negative
   ###### FP = 예측 값을 Positive 값으로 예측했고 실제 값이 Negative
   ###### FN = 예측 값을 Negative 값으로 예측했고 실제 값이 Positive
   ###### TP = 예측 값을 Positive 값으로 예측했고 실제 값이 Positive
   
 이렇게 4개 정보를 통해 분류의 성능을 측정할 수 있다. 오차행렬를 통해 성능을 측정하는 방법으로는 정확도, 정밀도, 재현율, F1 score를 알 수 있다.

   ###### 정확도 = 예측 결과와 실제 값이 동일한 건수/전체 데이터 수 (TN + TP) / (TN + FP + FN + TP)
     ###### 정확도는 성능을 평가하는 데 좋은 지표이지만 비대칭한 데이터 세트에서는 수치적인 판단 오류를 일으킬 수 있다.
   ###### 정밀도 = 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율 TP/ (FP + TP)
     ###### Positive 예측 성능을 더욱 정밀하게 측정하기 위한 평가 지표로 양성 예측도라고 불린다.
   ###### 재현율 = 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율 TP/ (FN + TP)
     ###### 민감도라고 불린다.

이진 분류 모델에서 업무 특성에 따라 특정 평가 지표가 더 중요한 지표로 간주될 수 있다. 재현율은 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 큰 영향이 발생하는 의료 분야에서 중요하다. 정밀도는 Negative 데이터를 Positive 데이터로 잘못 판단하게 되면 큰 영향이 발생하는 스팸 메일 판정에서 중요하다.

   ###### F1 score = 정밀도와 재현율을 결합한 지표
     ###### F1 점수가 높다는 것은 모델이 정확하게 예측하면서도 많은 긍정적인 사례를 잡아내고 있다는 의미
   ###### ROC 곡선과 AUC = 이진 분류에서 예측 성능 측정에서 중요하게 사용되는 지표
     ###### ROC 곡선은 FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선이다. FPR을 X축으로, TPR을 Y 축으로 잡으면 FPR의 변화에 따른 TPR의 변화가 곡선 형태로 나타난다.
     ###### 분류의 성능 지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC 값으로 결정한다.
     ###### AUC(Area Under Curve) 값은 ROC 곡선 밑의 면적을 구한 것으로서 일반적으로 1에 가까울 수록 좋은 수치이다.
     
오차행렬과 정확도, 재현율, 정밀도, F1 score, ROC 곡선과 AUC를 설명한 이유는 앞서 설명했듯 Santander Customer Satisfaction 대회가 ROC 곡선의 아래 면적 즉, AUC를 평가 지표로 하기 때문이다. 뿐만 아니라 이전 tatinic data에서도 사용했지만 따로 설명하지 않았기 때문에 간단하게 설명했다.
