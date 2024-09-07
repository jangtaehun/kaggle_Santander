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
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```
분류 모델의 성능을 평가하기 위해 공통 코드로 위와 같이 오차행렬, 정확도, 재현율, 정밀도, F1 score, ROC 곡선과 AUC를 평가하기 위해 다음과 같이 코드를 작성했다. 공통 코드는 모델을 사용했을 때마다 성능을 평가하기 위해 사용할 것으로 함수화 했다. 각각 설명은 다음과 같다.

1. 오차행렬
혼동행렬(confusion matrix)라고도 부른다. 학습된 분류 모델이 예측을 수행하면서 얼마나 혼동(혼란)하고 있는지 보여주는 지표이다. 따라서 이진 분류에서 예측 오류가 얼마인지, 어떤 유형의 예측 오류가 발생하고 있는지 보여주는 지표이다.
![image](https://github.com/user-attachments/assets/aa79ad40-6ba9-4129-9545-ecb01fe83cb0)

위와 같이 총 4개 분면을 통해 분류 모델 예측 성능을 보여주는 지표이다. TN, FP, FN, TP가 있으며 각각 다음과 같은 지표를 보여준다.

 * TN = 예측 값을 Negative 값으로 예측했고 실제 값이 Negative
 * FP = 예측 값을 Positive 값으로 예측했고 실제 값이 Negative
 * FN = 예측 값을 Negative 값으로 예측했고 실제 값이 Positive
 * TP = 예측 값을 Positive 값으로 예측했고 실제 값이 Positive
   
 이렇게 4개 정보를 통해 분류의 성능을 측정할 수 있다. 오차행렬를 통해 성능을 측정하는 방법으로는 정확도, 정밀도, 재현율, F1 score를 알 수 있다.

   ###### 정확도 = 예측 결과와 실제 값이 동일한 건수/전체 데이터 수 (TN + TP) / (TN + FP + FN + TP)
      정확도는 성능을 평가하는 데 좋은 지표이지만 비대칭한 데이터 세트에서는 수치적인 판단 오류를 일으킬 수 있다.
   ###### 정밀도 = 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율 TP/ (FP + TP)
      Positive 예측 성능을 더욱 정밀하게 측정하기 위한 평가 지표로 양성 예측도라고 불린다.
   ###### 재현율 = 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율 TP/ (FN + TP)
      민감도라고 불린다.

이진 분류 모델에서 업무 특성에 따라 특정 평가 지표가 더 중요한 지표로 간주될 수 있다. 재현율은 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 큰 영향이 발생하는 의료 분야에서 중요하다. 정밀도는 Negative 데이터를 Positive 데이터로 잘못 판단하게 되면 큰 영향이 발생하는 스팸 메일 판정에서 중요하다.

   ###### F1 score = 정밀도와 재현율을 결합한 지표
      F1 점수가 높다는 것은 모델이 정확하게 예측하면서도 많은 긍정적인 사례를 잡아내고 있다는 의미
   ###### ROC 곡선과 AUC = 이진 분류에서 예측 성능 측정에서 중요하게 사용되는 지표
      ROC 곡선은 FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선이다. FPR을 X축으로, TPR을 Y 축으로 잡으면 FPR의 변화에 따른 TPR의 변화가 곡선 형태로 나타난다.
      분류의 성능 지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC 값으로 결정한다.
      AUC(Area Under Curve) 값은 ROC 곡선 밑의 면적을 구한 것으로서 일반적으로 1에 가까울 수록 좋은 수치이다.
     
오차행렬과 정확도, 재현율, 정밀도, F1 score, ROC 곡선과 AUC를 설명한 이유는 앞서 설명했듯 Santander Customer Satisfaction 대회가 ROC 곡선의 아래 면적 즉, AUC를 평가 지표로 하기 때문이다. 뿐만 아니라 이전 tatinic data에서도 사용했지만 따로 설명하지 않았기 때문에 간단하게 설명했다.

---

   #### 1. Santander Customer Satisfaction data set에 대한 기본적인 정보
Santander Customer Satisfaction data는 아래 사진과 같이 모든 feature가 개인정보를 이유로 feature의 이름이 모두 익명처리 되어있다.
![image](https://github.com/user-attachments/assets/3e4b447e-91b2-487d-931f-4c78b6b60c96)

따라서 어떤 의미를 가진 것인지 추정할 수 없다. 따라서 상관관계에 대한 분석을 진행할 것이며, 이전에 진행했던 titanic data에 대한 분석을 했던 것만큼 자세한 분석은 진행하는 데 한계가 있다.
```
train_df.info()
```
데이터에 대한 정보를 보면 706,020개의 row가 있고, 371개의 column이 있다. data type의 경우 float64, int64 각각 111개, 260개로 object type은 없다. 
```
rain_df.describe()
```
요약된 정보를 좀 더 자세하게 알 수 있다. 모든 columns가 float, int type이기에 생략된 부분은 없을 것이다. 하지만 column이 많아 모든 행이 출력되지 않아(설정으로 모든 column이 나오게할 수 있다.) 전부를 확인할 수 없다.
![image](https://github.com/user-attachments/assets/7036aefa-8fa5-497a-b333-da723c632431)

출력된 내용을 보면 이상한 부분이 var3이다. var3의 경우 min 값이 -999999로 나온다. var3이 무엇인지는 몰라도 -999999는 충분히 의심할 수 있는 값이다. 아마 NaN인 값을 -999999로 대체했을 가능성이 있다. 따라서 var3과 같이 이상치가 있는 컬럼이 더 존재할 수 있기 때문에 확인이 필요하다.

   #### 2. Feature 분석
먼저 feature의 수가 많으며, 정확히 어떤 데이터인지 확인이 불분명한 데이터 이기에 필요 없는 데이터와 필요한 데이터를 구분해야 한다. 따라서 모든 값이 NaN값인 컬럼은 drop하는 작업을 먼저 하겠다.
```
all_nan_columns = train_df.columns[train_df.isna().all()].tolist()
print(f"모든 값이 NaN인 컬럼 개수: {len(all_nan_columns)}")

train_df.drop(columns=all_nan_columns, inplace=True, axis=1)
test_df.drop(columns=all_nan_columns, inplace=True, axis=1)
```
위의 코드를 실행하면 아래와 같이 출력된다. Santander에서 제공한 데이터는 모든 값이 NaN값인 컬럼은 존재하지 않는다.
```
모든 값이 NaN인 컬럼 개수: 0
```
NaN 값 확인이 끝났으면 다음으로 모든 값이 같은 즉, 특정 컬럼에서의 값이 모두 같은 컬럼을 drop하는 작업을 하겠다. 예를 들어 어떤 컬럼의 모든 값이 0 또는 1인 컬럼을 drop하는 것이다.
```
unique_one_columns = [col for col in train_df.columns if train_df[col].nunique() == 1]
print(f'고유값이 1인 컬럼 개수: {len(unique_one_columns)}')
```
위의 코드를 실행하면 아래와 같이 출력된다. Santander에서 제공한 데이터는 고유값이 1인 컬럼이 34개나 존재하고 있다.  이 컬럼들은 나중에 제거를 할 것이다.
```
고유값이 1인 컬럼 개수: 34
```
모든 값이 같은 컬럼을 drop하는 이유는 다음과 같다.
* 모든 샘플에서 동일한 값을 가지므로, 이 컬럼은 학습 데이터에서 어떠한 예측 정보도 제공하지 못 하기 때문이다.
* 불필요한 컬럼을 제거해 모델의 복잡성을 줄일 수 있다.
* 불필요한 컬럼이 많을 경우, 모델이 의미 없는 패턴을 학습하는 과적합 위험이 증가할 수 있기 때문이다.
* 데이터의 크기가 줄어들기 때문에 저장 공간과 처리 시간이 절약되기 때문에 대규모 데이터셋을 다룰 때 매우 중요하다.
위와 같은 이유로 고유값이 1인 컬럼 즉, 모든 값이 같은 컬럼을 drop하는 것이다.
![image](https://github.com/user-attachments/assets/a07a2207-cb60-403c-9aec-24f609277ed2)

또한, 위의 describe() 메서드를 통해 얻은 결과에서 mean 값을 살펴보면 같은 값을 가진 컬럼이 존재하는 것을 확인할 수 있다. ind_var13_medio_0와 ind_var13_medio를 보면 mean 값이 같다. 즉, 두 개의 컬럼이 이름도 비슷하고 값도 같다. 따라서 이런 부분에 대해서도 처리가 필요하다. 이 부분 역시 나중에 제거를 할 것이다.
   
   #### 3. 이상치 탐색
이상치 제거는 데이터에서 비정상적으로 크거나 작은 값, 즉 다른 데이터와 현저히 차이가 나는 값을 제거하거나 처리하는 것으로 데이터의 왜곡을 방지하기 위해 필요한 작업이다. 따라서 이상치를 탐색하고 처리하겠다.

위에서 아래와 같이 describe() 메서드를 통해 요약된 정보를 확인했다. 이때 주목할 부분이 var3으로 min 값이 -999999로 되어있다. 따라서 Santander에서 제공한 데이터는 이상치가 포함된 값으로 이상치를 탐색한 후 이상치에 대한 적절한 처리를 해야 한다.
![image](https://github.com/user-attachments/assets/a5f5b863-a837-4b55-b67b-1be788f681b4)

먼저 var3에 대한 값을 살펴보겠다.
```
train_df[train_df['var3']==-999999]
```
위의 코드를 실행해 아래와 같은 dataframe을 출력할 수 있다. var3의 값이 -999999인 값을 가진 row는 116개가 있다. 이 row는 다른 값 역시 이상치를 가질 확률이 있을 수 있다. 따라서 다른 컬럼에도 이상치가 있는지 여부를 확인하는 데 좋은 정보를 줄 수 있다.
![image](https://github.com/user-attachments/assets/36b948e0-1ce5-4472-ba28-8155cf4064f0)
![image](https://github.com/user-attachments/assets/d95fadd2-7324-4f6d-9055-e348b81d05dc)

var3이 -999999인 것만 따로 출력한 dataframe을 보면 0이 상당히 많다. 뿐만 아니라 var38의 경우 같은 값을 가진 숫자가 많다. 따라서 0의 갯수에 따른 처리와 var38에 대한 처리도 필요하다.

   ##### var3
1. var3에 대해 -999999를 가장 많은 값을 가지고 있는 2로 대체 (최빈값으로 대체)
아래의 코드를 통해 얻은 결과를 보면 va3 컬럼에서 가장 많은 값을 가지고 있는 값은 2이다. 따라서 2로 -999999를 대체하는 방법을 적용해 볼 것이다.
```
train_df['var3'].value_counts()
```
![image](https://github.com/user-attachments/assets/6279b6a6-0833-4895-b471-0db92d7ae21c)

2. va3에 대해 -999999를 NaN 값에 대한 처리로 예상하고 있기 때문에 값을 -1로 대체 (고정값 대체)
3. var3의 -999999를 새로운 열로 만들어 추가 (NaN 값 자체를 특성화)

   ##### var38
필자는 var38에서 117310.979016494의 값이 var38에서 NaN 값을 평균으로 대체한 값이라 생각한다. 그 이유는 다음과 같다. 필자는 마지막에 있는 vr38이 고객의 자산이지 않을까 조심스럽게 예측하고 있다. 이때 아래의 코드를 통해 얻은 결과를 보면 자산이 같은 값이 14868개라 보기엔 이상하다. var3에 대해서는 -999999가 이상치라 구분이 갔지만 var38에 대해서는 57736개의 nunique가 있는데 유독 하나의 값에 몰려 있다는 것은 이상하다 보기 때문이다. 따라서 이 부분에 대해서도 처리를 해보려고 한다.
```
train_df['var38'].value_counts()
```
![image](https://github.com/user-attachments/assets/7cf52567-c5fd-4034-b3c6-46248a9d36ec)
1. va38에 대해 117310.979016494를 NaN 값에 대한 처리로 예상하고 있기 때문에 값을 -1로 대체 (고정값 대체)
2. var38의 117310.979016494를 새로운 열로 만들어 추가 (NaN 값 자체를 특성화)
3. var38의 117310.979016494를 그대로 사용
이렇게 총 6가지의 방법으로 테스트를 해보려고 한다.

   #### 4. Data cleaning
노이즈 제거는 데이터에서 불필요하거나 무작위적인 변동을 제거하여 데이터의 신호를 명확하게 하고, 분석 또는 모델링의 정확성을 높이는 것으로 데이터의 본질적인 신호를 더 잘 이해하거나 예측하기 위해 필요한 작업이다. 따라서 노이즈를 탐색하고 처리하겠다.

다음으로 같은 같은 피처(특징)를 가진 행이 서로 다른 클래스 레이블(TARGET)을 가지는 경우를 찾아서 처리하는 작업을 진행하려 한다. 먼저 ID, TARGET 컬럼을 제거 및 분리하는 작업을 하겠다.
```
train_df.drop(['ID'], axis=1, inplace=True)
test_df.drop(['ID'], axis=1, inplace=True)

y = train_df['TARGET']
X = train_df.drop('TARGET', axis=1)
```
ID와 TARGET을 제거한 다음 위에서 확인했던 고유값이 1인 컬럼을 제거할 것이다. 아래의 코드를 이용해 제거를 하면 총 34의 컬럼을 제거해 남은 컬럼은 337개의 컬럼만 남는다.
```
unique_one_columns = [col for col in train_df.columns if train_df[col].nunique() == 1]
print(f'고유값이 1인 컬럼 개수: {len(unique_one_columns)}')

train_df.drop(columns=unique_one_columns, inplace=True, axis=1)
test_df.drop(columns=unique_one_columns, inplace=True, axis=1)
```
이번에도 위에서 확인했듯 같은 값을 가진 컬럼을 제거할 것이다. 아래와 같은 코드를 실행하면 다음과 같이 결과를 얻을 수 있다. 
```
duplicate_columns = []
columns = train_df.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if train_df[columns[i]].equals(train_df[columns[j]]):
            duplicate_columns.append((columns[i], columns[j]))

for col1, col2 in duplicate_columns:
    print(f"{col1} == {col2}")
    train_df.drop([col2], axis=1, inplace=True)
    test_df.drop([col2], axis=1, inplace=True)
```
![image](https://github.com/user-attachments/assets/7960ab5d-3dff-4456-a01e-fc4bb7604fe5)

위에 있는 출력 값들이 모두 서로 같은 값을 가진 컬럼이다. var6_0, var29_0, var6, var29가 서로 같은 값을 가진다는 것을 제외하면 모든 컬럼이 비슷한 이름을 가진 것을 알 수 있다. 이 컬럼들을 제거하면 29개의 컬럼이 추가적으로 삭제되어 308개의 column이 남는다.
```
train_with_target = pd.concat([X, y], axis=1)
duplicates = train_with_target.duplicated(keep=False)
duplicates_with_different_target = duplicates & (train_with_target.groupby(list(X.columns))['TARGET'].transform('nunique') > 1)

noise = train_with_target[duplicates_with_different_target]
cleaned_train = train_with_target[~duplicates_with_different_target]

X = cleaned_train.drop('TARGET', axis=1)
y = cleaned_train['TARGET']
```
이번에는 column이 아닌 row에서 data cleaning을 하려고 한다. noise 즉, 중복된 피처 값을 가진 데이터 중에서 타겟 값이 다른 데이터를 확인하고 제거하고자 한다. 위의 코드를 출력해보면 아래와 같다. 총 2435개의 행이 중복된 것이다.
![image](https://github.com/user-attachments/assets/a89de785-8121-4456-a369-ba1529fbfc18)

결과적으로 X에서 제거된 행의 개수는 2435의 행이 제거 되어 총 73290개의 행이 남는다. 따라서 동일한 행을 가지지만 다른 타겟 값을 가지는 행이 매우 많았다는 것을 알 수 있다. 다음으로 위에서 언급했듯 0과 var3의 -999999 값을 최빈값인 2로 대체한 후 var38의 117310.979016494를 -1로 대체하는 작업을 진행하겠다.

   #### 5. Feature Engineering
0에 대한 처리를 진행하겠다. 지금까지 확인했듯이 Santander에서 제공한 Santander Customer Satisfaction 데이터는 0이 굉장히 많다. 따라서 이 부분에 대해서도 적절한 처리가  필요하다. 필자는 각 행(row)에서 0의 갯수를 새로운 컬럼으로 저장할 것이다. 아래의 코드를 실행하면 다음과 같이 결과가 나온다.
```
train_df['count_0'] = (train_df == 0).sum(axis=1)
test_df['count_0'] = (test_df == 0).sum(axis=1)
```
![image](https://github.com/user-attachments/assets/6273c3a8-eead-400c-9299-342ba2683a94)

위의 코드를 통해 var3과 var38에 대해 처리를 했다.

다음으로 isolationforest를 사용해 이상치 탐지를 하겠다. isolationforest는비지도학습 기반의 이상 탐지 알고리즘이다. 비지도 학습 중에서 이상치를 탐지하는 데 강력한 알고리즘이다. Santander Customer Satisfaction data는 이상치를 탐지하기 어려운 데이터라 모델을 통해 이상치를 제거했다.
```
from sklearn.ensemble import IsolationForest
import plotly.express as px 

# 비지도학습 기반의 이상 탐지 알고리즘
clf = IsolationForest(
    n_estimators=50, 
    max_samples=50, 
    contamination=float(0.004), 
    max_features=1.0, 
    bootstrap=False, 
    n_jobs=-1, 
    verbose=0)

# 모델 학습
clf.fit(X)
pred = clf.predict(X)

# 예측 결과를 데이터프레임에 추가
X['label'] = pred

# 이상치 데이터 추출 / 1=정상, -1=이상치
outliers = X.loc[X['label'] == -1]
outlier_index = list(outliers.index)

# 이상치와 정상치 개수 출력
print(X['label'].value_counts()) 

# 이상치를 제외한 데이터 추출
X = X.loc[X['label'] != -1]
X = X.drop(columns=['label'])  # 'label' 열 제거

# y에서도 이상치 인덱스 제거
y = y.drop(outlier_index)
```
다음으로 동일한 행을 가지지만 다른 타겟 값을 가지는 행이 있기 때문에 데이터를 5개로 나눈 후 모델을 학습해 노이즈 데이터에 대해 TARGET 값을 예측하겠다. 이유는 다음과 같다. 데이터를 나누어 여러 모델을 학습시키는 것은 모델의 안정성과 일반화 성능을 높이고, 데이터의 다양성을 충분히 반영하여 과적합을 방지할 수 있기 때문이다.
```
import optuna
import xgboost as xgb
from scipy.sparse import csr_matrix

train_parts = np.array_split(X, 5)
train_y_parts = np.array_split(y, 5)

def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        'alpha': trial.suggest_float('reg_alpha', 1.0, 20.0),
        'lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    f1_scores = []

    for train_part, train_y_part in zip(train_parts, train_y_parts):
        dtrain = xgb.DMatrix(csr_matrix(train_part.values), label=train_y_part)   
        bst = xgb.train(param, dtrain, num_boost_round=500)
        y_val_pred = (bst.predict(dtrain) > 0.5).astype(int)
        f1 = f1_score(train_y_part, y_val_pred)
        f1_scores.append(f1)

    return np.mean(f1_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best hyperparameters:", best_params)
```
```
param = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': best_params['learning_rate'],
    'max_depth': best_params['max_depth'],
    'min_child_weight': best_params['min_child_weight'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'scale_pos_weight': best_params['scale_pos_weight'],
    'alpha': best_params['reg_alpha'],
    'lambda': best_params['reg_lambda'],
    'n_estimators': best_params['n_estimators']
}

bst_models = []

# 5개의 파트에 대해 학습
for train_part, train_y_part in zip(train_parts, train_y_parts):
    dtrain = xgb.DMatrix(csr_matrix(train_part.values), label=train_y_part)
    bst = xgb.train(param, dtrain, num_boost_round=500)
    bst_models.append(bst)

# 노이즈 데이터 예측
noise['TARGET'] = 0  # 초기값 설정
dnoise = xgb.DMatrix(csr_matrix(noise.drop('TARGET', axis=1).values))
noise_preds = np.mean([bst.predict(dnoise) for bst in bst_models], axis=0)

# 0.5 이상이면 1로, 그렇지 않으면 0으로 설정
noise['TARGET'] = (noise_preds >= 0.5).astype(int)

# noise 데이터와 원래 train 데이터 병합
X = pd.concat([X, noise.drop('TARGET', axis=1)])
y = pd.concat([y, noise['TARGET']])

# 최종 데이터 크기 출력
print(f"Final train shape: {X.shape}")
print(f"Final train_y shape: {y.shape}")
```
위의 코드와 같이 진행하면 noise로 분류되어 삭제되었던 부분의 TARGET을 새롭게 예측해 isolationforest로 제거한 이상치 행을 제외한 75725개의 행만 남는다. 추가적으로 var15에 대해서도 추가적인 분석을 진행하던 중 특정 패턴을 발견했다. 
```
var15_values_when_target_1 = train_df[train_df['TARGET'] == 1]['var15']

unique_var15_values = np.sort(var15_values_when_target_1.unique())
unique_var15_values
```
위와 같이 코드를 입력하고 실행했을 때 아래와 같이 출력이 된다. var15는 5부터 값이 있는 것으로 23보다 작으면 모든 값이 0이라는 것이다. 따라서 var15가 23보다 작으면 0으로 하드코딩을 할 수 있다.
![image](https://github.com/user-attachments/assets/9f0b90c9-88b0-4921-a636-8cb77192a3fb)

---

#### 모델 학습
