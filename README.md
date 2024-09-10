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
    * noise 처리
    * Feature Engineering
3. 모델 학습
  * XGBoost
  * LightGBM
  * CatBoost
  * Ensemble - Voting
4. 결론
  * 한계점

---

### 문제에 대한 정보 수집
   #### 1. 문제 정의
Santander Customer Satisfaction에 대한 data는 고객의 만족도를 개선하기 위해 kaggle에 데이터를 제공해주었다. 따라서 Kaggle에서 고객의 정보를 토대로 Santander 은행이 제공하는 서비스에 불만족을 느끼는 고객을 식별하는 대회가 2016년에 진행되었다.

train.csv을 기반으로 적절한 EDA를 진행한 후 test.csv의 데이터를 이용해 예측한 후 결과를sample_submission.csv와 결합한 후 제출하고 제출하면 된다.

![image](https://github.com/user-attachments/assets/5026c767-5401-4087-a83c-8baaa8955859)

Santander Customer Satisfaction data는 위와 같이 모든 feature가 개인정보를 이유로 feature의 이름이 모두 익명처리 되어있다. 따라서 어떤 속성인지 추정할 수 없다.

TARGET으로는 1은 불만족, 0은 만족한 고객을 나타내며, 고객의 만족도를 예측하는 문제이다. 뿐만 아니라 평가지표로는 다음과 같이 공지되어 있다.

![image](https://github.com/user-attachments/assets/a883b253-d187-4af2-aca2-dcd435bde71e)

<www.kaggle.com/competitions/santander-customer-satisfaction/overview/evaluation>

ROC 곡선의 아래 면적을 기준으로 평가된다. 따라서 모델의 예측 성능을 AUC로 측정하며, 높은 AUC를 얻는 것이 대회에서 좋은 성적을 얻는 데 중요하다.

---

   #### 2. 분석 대상에 대한 이해
Santander 은행은 스페인 산탄데르에 1867년에 설립되 유럽 최대 기업 및 은행이다. Santander 은행은 다른 세계적인 은행과 다른 특징이 있다. 대형 은행들은 투자금융 분야 규모가 크다. 하지만 Santander 은행은 수익의 큰 부분이 소매금융에서 나온다. 즉, 금융기관인 Santander 은행이 개인에게 금융 서비스를 제공하는 것이 수익의 큰 부분이다.

소매금융에서 나오는 수익이 크기 때문에 Santander 은행은 고객의 만족, 불만족에 큰 관심을 가지게 된 것으로 kaggle에 feature 이름이 익명처리된 data를 제공한 것이다.

고객을 대상으로 하는 많은 분야에서 고객 만족도는 성공의 중요한 척도이다. 기업에 불만족을 느끼는 고객은 더이상 고객으로 머물지 않는다. 하지만 대부분의 고객은 기업에서 제공하는 서비스에 대한 불만족을 떠나기 전까지 혹은 떠난 후에도 거의 표출하지 않는다.

따라서 Santander Bank는 kaggle에 고객에 대한 데이터를 제공하면서 초기 단계에서 불만족한 고객을 식별하는 데 도움을 요청했다. 즉, Santander Bank는 고객이 떠나기 전에 고객의 만족을 개선하기 위한 선제적인 조치를 취하고자 한 것이다.

Santander Customer Satisfaction data는 앞에서 말했듯 수백 개의 익명화된 특징을 사용하여 고객이 은행 경험에 만족하는지 불만족하는지를 예측하고자 한다. 필자는 이러한 특징으로 인해 Santander Customer Satisfaction 데이터를 마냥 쉬운 난이도로 보고있지 않다.

---

### Santander Customer Satisfaction data set을 이용한 EDA
   #### 1. 공통 코드
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
   #### 2. 분석
   ##### 1. Santander Customer Satisfaction data set에 대한 기본적인 정보
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

   ##### 2. Feature 분석
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
   
   ##### 3. 이상치 탐색
이상치 제거는 데이터에서 비정상적으로 크거나 작은 값, 즉 다른 데이터와 현저히 차이가 나는 값을 제거하거나 처리하는 것으로 데이터의 왜곡을 방지하기 위해 필요한 작업이다. 따라서 이상치를 탐색하고 처리하겠다.

위에서 아래와 같이 describe() 메서드를 통해 요약된 정보를 확인했다. 이때 주목할 부분이 var3으로 min 값이 -999999로 되어있다. 따라서 Santander에서 제공한 데이터는 이상치가 포함된 값으로 이상치를 탐색한 후 이상치에 대한 적절한 처리를 해야 한다.
![image](https://github.com/user-attachments/assets/a5f5b863-a837-4b55-b67b-1be788f681b4)

먼저 var3에 대한 값을 살펴보겠다.
```
train_df[train_df['var3']==-999999]
```
위의 코드를 실행해 아래와 같은 dataframe을 출력할 수 있다. var3의 값이 -999999인 값을 가진 row는 116개가 있다. 이 row는 다른 값 역시 이상치를 가질 확률이 있을 수 있다. 따라서 다른 컬럼에도 이상치가 있는지 여부를 확인하는 데 좋은 정보를 줄 수 있다.
![image](https://github.com/user-attachments/assets/36b948e0-1ce5-4472-ba28-8155cf4064f0)
![image](https://github.com/user-attachments/assets/4b3430a4-054b-47b8-8b96-a9aca44ef842)

var3이 -999999인 것만 따로 출력한 dataframe을 보면 0이 상당히 많다. 뿐만 아니라 var38의 경우 같은 값을 가진 숫자가 많다. 따라서 0의 갯수에 따른 처리와 var38에 대한 처리도 필요하다.

###### var3
1. var3에 대해 -999999를 가장 많은 값을 가지고 있는 2로 대체 (최빈값으로 대체)
아래의 코드를 통해 얻은 결과를 보면 va3 컬럼에서 가장 많은 값을 가지고 있는 값은 2이다. 따라서 2로 -999999를 대체하는 방법을 적용해 볼 것이다.
```
train_df['var3'].value_counts()
```
![image](https://github.com/user-attachments/assets/6279b6a6-0833-4895-b471-0db92d7ae21c)

2. va3에 대해 -999999를 NaN 값에 대한 처리로 예상하고 있기 때문에 값을 -1로 대체 (고정값 대체)
3. var3의 -999999를 새로운 열로 만들어 추가 (NaN 값 자체를 특성화)

###### var38
필자는 var38에서 117310.979016494의 값이 var38에서 NaN 값을 평균으로 대체한 값이라 생각한다. 그 이유는 다음과 같다. 필자는 마지막에 있는 vr38이 고객의 자산이지 않을까 조심스럽게 예측하고 있다. 이때 아래의 코드를 통해 얻은 결과를 보면 자산이 같은 값이 14868개라 보기엔 이상하다. var3에 대해서는 -999999가 이상치라 구분이 갔지만 var38에 대해서는 57736개의 nunique가 있는데 유독 하나의 값에 몰려 있다는 것은 이상하다 보기 때문이다. 따라서 이 부분에 대해서도 처리를 해보려고 한다.
```
train_df['var38'].value_counts()
```
![image](https://github.com/user-attachments/assets/7cf52567-c5fd-4034-b3c6-46248a9d36ec)
1. va38에 대해 117310.979016494를 NaN 값에 대한 처리로 예상하고 있기 때문에 값을 -1로 대체 (고정값 대체)
2. var38의 117310.979016494를 새로운 열로 만들어 추가 (NaN 값 자체를 특성화)
3. var38의 117310.979016494를 그대로 사용
이렇게 총 6가지의 방법으로 테스트를 해보려고 한다.

##### 4. Data cleaning
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

결과적으로 X에서 제거된 행의 개수는 2435의 행이 제거 되어 총 73290개의 행이 남는다. 따라서 동일한 행을 가지지만 다른 타겟 값을 가지는 행이 매우 많았다는 것을 알 수 있다. 다음으로 위에서 언급했듯 0과 var3의 -999999 값을 최빈값인 2로 대체한 후 var38의 117310.979016494를 -1로 대체하는 작업을 진행하겠다.
```
X['var3'].replace(-999999, 2, inplace=True)
test_df['var3'].replace(-999999, 2, inplace=True)

X.loc[np.isclose(X['var38'], 117310.979016), 'var38'] = -1
test_df.loc[np.isclose(test_df['var38'], 117310.979016), 'var38'] = -1
```
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

   ##### 5. noise 처리
다음으로 동일한 행을 가지지만 다른 타겟 값을 가지는 행이 있기 때문에 데이터를 5개로 나눈 후 모델을 학습해 노이즈 데이터에 대해 TARGET 값을 예측하겠다. 이유는 다음과 같다. 데이터를 나누어 여러 모델을 학습시키는 것은 모델의 안정성과 일반화 성능을 높이고, 데이터의 다양성을 충분히 반영하여 과적합을 방지할 수 있기 때문이다.
아래는 LGBMClassifier를 통해 noise에 대한 TARGET 값을 예측한 것이다.
```
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from scipy.sparse import csr_matrix

train_parts = np.array_split(X, 5)
train_y_parts = np.array_split(y, 5)

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    f1_scores = []
    
    for train_part, train_y_part in zip(train_parts, train_y_parts):
        lgb_model = LGBMClassifier(**param, random_state=42)
        lgb_model.fit(train_part, train_y_part)
        
        y_val_pred = lgb_model.predict(train_part)
        f1 = f1_score(train_y_part, y_val_pred)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params


best_lgb_model = LGBMClassifier(**best_params, random_state=42)
bst_models = []

for train_part, train_y_part in zip(train_parts, train_y_parts):
    best_lgb_model.fit(train_part, train_y_part)
    bst_models.append(best_lgb_model)

noise['TARGET'] = 0
noise_preds = np.mean([model.predict(noise.drop('TARGET', axis=1)) for model in bst_models], axis=0)

noise['TARGET'] = (noise_preds >= 0.5).astype(int)

X = pd.concat([X, noise.drop('TARGET', axis=1)])
y = pd.concat([y, noise['TARGET']])
```
noise 데이터에 대해 5개의 모델을 사용하여 타겟 값을 예측한 후 각 모델의 예측 값을 평균 내어 noise_preds에 저장하고, 그 값이 0.5 이상이면 1, 그렇지 않으면 0으로 타겟 값을 설정한 것이다.

위의 코드와 같이 진행하면 noise로 분류되어 삭제되었던 부분의 TARGET을 새롭게 예측해 isolationforest로 제거한 이상치 행을 제외한 75725개의 행만 남는다. 

   ##### 6. Feature Engineering
마지막으로 0에 대한 처리를 진행하겠다. 지금까지 확인했듯이 Santander에서 제공한 Santander Customer Satisfaction 데이터는 0이 굉장히 많다. 따라서 이 부분에 대해서도 적절한 처리가  필요하다. 필자는 각 행(row)에서 0의 갯수를 새로운 컬럼으로 저장할 것이다. 아래의 코드를 실행하면 다음과 같이 결과가 나온다.
```
train_df['count_0'] = (train_df == 0).sum(axis=1)
test_df['count_0'] = (test_df == 0).sum(axis=1)
```
![image](https://github.com/user-attachments/assets/6273c3a8-eead-400c-9299-342ba2683a94)

추가적으로 var15에 대해서도 추가적인 분석을 진행하던 중 특정 패턴을 발견했다. 아래의 코드를 통해 확인할 수 있다.
```
var15_values_when_target_1 = train_df[train_df['TARGET'] == 1]['var15']

unique_var15_values = np.sort(var15_values_when_target_1.unique())
unique_var15_values
```
위와 같이 코드를 입력하고 실행했을 때 아래와 같이 출력이 된다. var15는 5부터 값이 있는 것으로 23보다 작으면 모든 값이 0이라는 것이다. 따라서 var15가 23보다 작으면 0으로 하드코딩을 할 수 있다.
![image](https://github.com/user-attachments/assets/9f0b90c9-88b0-4921-a636-8cb77192a3fb)

---

### 모델 학습
RandomUnderSampler() 클래스를 이용해 데이터의 불균형을 해결하기 위한 코드이다. Santander Customer Satisfaction data는 불균형한 데이터이다. 따라서 이를 처리하기 위한 방법이 필요하다. 필자는 오버샘플링, 언더샘플링, 하이브리드 샘플링에서 언더샘플링을 선택했다. 이유는 다음과 같다. 언더 샘플링은 데이터에서 빈도가 높은 클래스의 표본 수를 감소시켜 빈도가 적은 클래스와 비슷한 수준으로 맞추는 방법이다. 다수 클래스의 표본 수를 줄이면 모델이 소수 클래스도 잘 학습할 수 있도록 도와준다. 또한, 과다표현된 클래스의 데이터를 줄여 소수 클래스에 대한 재현율을 높이고 모델의 편향을 방지하는 데 효과적이기 때문에 사용했다. 하지만 데이터 손실의 위험이 있기 때문에 조심해야 한다.

먼저 모델을 학습하기 전에 데이터에 대한 처리를 먼저할 것이다. StandardScaler()를 통해 특성(Feature)의 값 범위를 표준화하거나 정규화하는 과정을 거치고 데이터를 언더 샘플링을 한 후 train 세트와 test 세트로 나눌 것이다. 이후 언더샘플링을 진행하고 train, test 세트로 나눌 것이다.
```
sc = StandardScaler()
X = sc.fit_transform(X)
test_df = sc.transform(test_df)
```
```
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold

X_resampled, y_resampled = RandomUnderSampler(random_state=42, sampling_strategy=0.3).fit_resample(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

   #### XGBoost
먼저 XGBoost를 사용할 것이다. 하이퍼파라미터 튜닝을 위해 optuna를 사용했다.
```
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score

def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),  # XGBoost에서는 num_leaves 대신 max_leaves가 있음. 그러나 생략 가능.
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    xgb_model = XGBClassifier(**param, random_state=42, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    
    y_val_pred = xgb_model.predict(X_val)
    
    f1 = f1_score(y_val, y_val_pred, pos_label=1) 
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best params: ", best_params)

best_xgb_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
best_xgb_model.fit(X_train, y_train)

y_val_pred = best_xgb_model.predict(X_val)

print(classification_report(y_val, y_val_pred))
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print(f"F1 Score: {f1_score(y_val, y_val_pred, pos_label=1)}")
print(f"Precision: {precision_score(y_val, y_val_pred, pos_label=1)}")  
print(f"Recall: {recall_score(y_val, y_val_pred, pos_label=1)}")
```

```
y_train_pred = best_xgb_model.predict(X_train)
y_train_pred_proba = best_xgb_model.predict_proba(X_train)[:, 1]

y_test_pred = best_xgb_model.predict(X_val)
y_test_pred_proba = best_xgb_model.predict_proba(X_val)[:, 1]

print("Train Data Evaluation:")
get_clf_eval(y_train, y_train_pred, y_train_pred_proba)
print("\nValidation Data Evaluation:")
get_clf_eval(y_val, y_test_pred, y_test_pred_proba)
```
다음과 같은 결과를 확인할 수 있다.
```
Train Data Evaluation:
오차 행렬
[[7461 1626]
 [ 314 2461]]
정확도: 0.8365, 정밀도: 0.6022, 재현율: 0.8868,    F1: 0.7173, AUC:0.9236

Validation Data Evaluation:
오차 행렬
[[1848  471]
 [ 122  525]]
정확도: 0.8001, 정밀도: 0.5271, 재현율: 0.8114,    F1: 0.6391, AUC:0.8751
```
다음으로 var15에 대해 23보다 작으면 0으로 변환하는 작업을 하겠다.
```
test_df = pd.DataFrame(test_df, columns=columns)

predict_santander_pred_xgb = best_xgb_model.predict(test_df)
test_df['TARGET'] = predict_santander_pred_xgb

test_y = test_df['TARGET']
test_X = test_df.drop(['TARGET'], axis=1)

test_df_original = sc.inverse_transform(test_X)
test_df_original = pd.DataFrame(test_df_original, columns=columns)

test_df_original['TARGET'] = test_y.values
test_df_original.loc[test_df_original['var15'] < 23, 'TARGET'] = 0

santander_submission_df['TARGET'] = test_df['TARGET']

# 결과를 CSV 파일로 저장
santander_submission_df.to_csv('santander_submission_lgbm.csv', index=False)
santander_submission_df
```
best 파라미터로 학습한 모델을 통해 위의 코드를 실행하면 다음과 같이 결과가 나온다.
![image](https://github.com/user-attachments/assets/fcf15518-f940-4182-8e85-a94edbe390ca)

좋은 점수를 보여준다.


   #### LightGBM
```
import optuna

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    lgb_model = LGBMClassifier(**param, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    y_val_pred = lgb_model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, pos_label=1) 
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

best_params = study.best_params
print("Best params: ", best_params)

best_lgb_model = LGBMClassifier(**best_params, random_state=42)
best_lgb_model.fit(X_train, y_train)

y_val_pred = best_lgb_model.predict(X_val)

y_train_pred = best_lgb_model.predict(X_train)
y_train_pred_proba = best_lgb_model.predict_proba(X_train)[:, 1]

y_test_pred = best_lgb_model.predict(X_val)
y_test_pred_proba = best_lgb_model.predict_proba(X_val)[:, 1]

get_clf_eval(y_train, y_train_pred, y_train_pred_proba)
get_clf_eval(y_val, y_test_pred, y_test_pred_proba)
```
결과는 아래와 같이 나온다. XGBoost가 재현율(Recall)과 F1 점수에서 더 좋은 성능을 보여주기 때문에, 특히 재현율이 중요한 문제(예: 질병 진단, 사기 탐지 등)에서는 XGBoost가 더 적합할 수 있다.
```
Train Data Evaluation:
오차 행렬
[[7468 1219]
 [ 674 1999]]
정확도: 0.8334, 정밀도: 0.6212, 재현율: 0.7478,    F1: 0.6787, AUC:0.8911

Validation Data Evaluation:
오차 행렬
[[1901  335]
 [ 176  428]]
정확도: 0.8201, 정밀도: 0.5609, 재현율: 0.7086,    F1: 0.6262, AUC:0.8672
```
다음으로 모델이 예측한 값에서 var15의 값이 23보다 작을 경우 0으로 바꾸는 작업을 할 것이다.
```
# scaling으로 dataframe에서 ndarray로 변환된 값을 다시 dataframe으로 변환
test_df = pd.DataFrame(test_df, columns=columns)

# test_df를 예측한 후 test_df의 TARGET 컬럼을 만든 후 값을 저장
# 이후 test_df를 X, y로 분리
predict_santander_pred_xgb = best_lgb_model.predict(test_df)
test_df['TARGET'] = predict_santander_pred_xgb
test_y = test_df['TARGET']
test_X = test_df.drop(['TARGET'], axis=1)

# test_df를 inverse_transform()을 이용해 scaling 전으로 되돌린다.
test_df_original = sc.inverse_transform(test_X)
test_df_original = pd.DataFrame(test_df_original, columns=columns)
test_df_original['TARGET'] = test_y.values

# test_df_original에서 var15의 값이 23보다 작으면 0으로 변경
test_df_original.loc[test_df_original['var15'] < 23, 'TARGET'] = 0

# submission_df에 test_df_original의 TARGET을 저장
santander_submission_df['TARGET'] = test_df_original['TARGET']
santander_submission_df.to_csv('santander_submission_lgbm.csv', index=False)
santander_submission_df
```
kaggle에 제출하면 다음과 같은 점수를 확인할 수 있다. 예상했듯이 XGBoost가 더 좋은 성능을 보여주고 있다.
![image](https://github.com/user-attachments/assets/749092ee-2b0b-43b0-b7b2-65bc7c573650)

   #### CatBoost
다음으로 진행할 모델은 CatBoost이다. CatBoost는 범주형 변수가 많은 데이터셋에서 탁월한 성능을 보여준다. 하지만 다양한 유형의 데이터에서도 높은 성능을 발휘하며, 과적합 방지, 병렬 처리 및 효율적인 메모리 사용으로 학습과 예측이 빠르다는 장점이 있다. 또한, 자동 하이퍼파라미터 튜닝이 있어 편리하다. 하지만 필자는 하이퍼파라미터 튜닝을 optuna를 통해 할 예정이다.
```
from catboost import CatBoostClassifier

def objective(trial):
    param = {
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),  # max_depth에 해당
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        'border_count': trial.suggest_int('border_count', 32, 255),  # colsample_bytree에 해당하는 역할
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0)
    }

    cat_model = CatBoostClassifier(**param, random_state=42, verbose=0)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50)

    y_val_pred = cat_model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred)
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best params: ", best_params)

best_cat_model = CatBoostClassifier(**best_params, random_state=42, verbose=0)
best_cat_model.fit(X_train, y_train)

y_val_pred = best_cat_model.predict(X_val)


y_train_pred = best_cat_model.predict(X_train)
y_train_pred_proba = best_cat_model.predict_proba(X_train)[:, 1]

y_test_pred = best_cat_model.predict(X_val)
y_test_pred_proba = best_cat_model.predict_proba(X_val)[:, 1]


print("Train Data Evaluation:")
get_clf_eval(y_train, y_train_pred, y_train_pred_proba)
print("\nValidation Data Evaluation:")
get_clf_eval(y_val, y_test_pred, y_test_pred_proba)
```
```
Train Data Evaluation:
오차 행렬
[[6439  994]
 [ 620 1615]]
정확도: 0.8331, 정밀도: 0.6190, 재현율: 0.7226,    F1: 0.6668, AUC:0.8838

Validation Data Evaluation:
오차 행렬
[[1578  285]
 [ 192  362]]
정확도: 0.8026, 정밀도: 0.5595, 재현율: 0.6534,    F1: 0.6028, AUC:0.8411
```
위와 같은 결과를 확인할 수 있다. XGBoost 보다는 과적합이 많이 해소된 것으로 보이지만 전체적으로 모델의 성능이 낮다. 특히, 정밀도가 낮은데, 이는 양성으로 예측한 것들 중 실제로 맞춘 비율이 낮다. 그럼에도 이번 데이터의 평가 지표인 AUC는 양호한 점수를 보여주고 있다. kaggle에 제출하면 아래와 같은 점수를 확인할 수 있다.
![image](https://github.com/user-attachments/assets/f5b68839-dba9-420a-9d4c-1ad40252be86)

   #### Ensemble
마지막으로 진행할 모델은 Ensemble 이다. 여러 개의 모델을 결합하여 하나의 모델보다 더 나은 성능을 얻는 기법이다. 이런 방법을 사용하면 개별 모델이 가지는 약점을 보완하고 예측의 안정성을 높이는 데 유리하다.

성능 향상: 개별 모델보다 더 높은 성능을 보일 수 있습니다.
안정성: 하나의 모델에서 발생할 수 있는 오류나 편향을 줄입니다.
유연성: 서로 다른 유형의 모델을 결합하여 더 복잡한 문제를 해결할 수 있습니다.
위와 같은 장점이 있으며, Bagging (배깅), Boosting (부스팅), Stacking (스태킹), Voting (보팅)이 있다. 이번에는 보팅 그 중에서도 소프트 보팅을 사용하려고 한다. 보팅은 여러 개의 모델을 학습한 후, 각 모델의 예측값을 투표 방식으로 결합하여 최종 예측을 도출하는 방식이다. 다수결 또는 가중치 기반 방식으로 최종 결과를 산출할 수 있다.

하드 보팅은 각 모델이 예측한 클래스(라벨) 중 다수결로 최종 클래스를 선택하며, 소프트 보팅은 모델들이 예측한 클래스 확률 값을 평균 내서 최종 클래스를 선택하는 것으로 확률 값이 반영되므로 더 정확한 예측이 가능할 수 있다.
```
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 모델 정의
decision_tree_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)
log_reg_clf = LogisticRegression(random_state=42)
adaboost_clf = AdaBoostClassifier(random_state=42)
lgbm_clf = LGBMClassifier(random_state=42, verbose=-1)
catboost_clf = CatBoostClassifier(random_state=42, verbose=0)
rf_clf = RandomForestClassifier(random_state=42)

# VotingClassifier를 사용한 소프트 보팅 모델 정의 (voting='soft')
voting_clf = VotingClassifier(
    estimators=[
        ('decision_tree', decision_tree_clf), 
        ('svm', svm_clf), 
        ('log_reg', log_reg_clf),
        ('adaboost', adaboost_clf), 
        ('lgbm', lgbm_clf), 
        ('catboost', catboost_clf), 
        ('rf', rf_clf)
    ], 
    voting='soft'  # 소프트 보팅 사용
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Soft Voting Classifier Accuracy: {accuracy:.4f}")

# 개별 모델의 성능 확인 (각 모델의 성능도 출력)
for clf in (decision_tree_clf, svm_clf, log_reg_clf, adaboost_clf, lgbm_clf, catboost_clf, rf_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"{clf.__class__.__name__} Accuracy: {accuracy_score(y_val, y_pred):.4f}")
```
```
Soft Voting Classifier Accuracy: 0.8481
LGBMClassifier Accuracy: 0.8415
CatBoostClassifier Accuracy: 0.8422
XGBClassifier Accuracy: 0.8394
RandomForestClassifier Accuracy: 0.8299
SVC Accuracy: 0.7952
```
각각 모델에 대해 위의 결과처럼 출력이 된다. 비슷한 점수대를 보여주지만 VotingClassifier가 개별 모델들보다 약간 더 높은 성능을 보이고 있으므로 잘 작동하고 있다고 볼 수 있다. 
```
y_train_pred = voting_clf.predict(X_train)
y_train_pred_proba = voting_clf.predict_proba(X_train)[:, 1]

y_test_pred = voting_clf.predict(X_val)
y_test_pred_proba = voting_clf.predict_proba(X_val)[:, 1]

print("Train Data Evaluation:")
get_clf_eval(y_train, y_train_pred, y_train_pred_proba)

print("\nValidation Data Evaluation:")
get_clf_eval(y_val, y_test_pred, y_test_pred_proba)
```
```
Train Data Evaluation:
오차 행렬
[[8464  282]
 [ 693 1962]]
정확도: 0.9145, 정밀도: 0.8743, 재현율: 0.7390,    F1: 0.8010, AUC:0.9758

Validation Data Evaluation:
오차 행렬
[[2069  148]
 [ 285  349]]
정확도: 0.8481, 정밀도: 0.7022, 재현율: 0.5505,    F1: 0.6172, AUC:0.8619
```

결과에 대한 평가는 다음과 같다. 이전에 학습했던 모델들에 비해 과적합이 다시 심해졌다. kaggle에 제출하면 다음과 같은 점수를 확인할 수 있다. Ensemble은 어떤 모델을 사용하냐에 따라 성능이 달라진다. 특히 Voting은 사용하는 개별 모델의 특성에 따라 성능이 크게 달라질 수 있기 때문에 모델 선택이 중요하다. 그 이유는 다음과 같다. 

성능 향상의 핵심은 모델의 다양성으로 Voting 앙상블은 서로 다른 특성을 가진 모델을 결합할 때 더 효과적이기 때문이다. 동일한 특성을 가진 모델들을 결합하면 성능 개선 효과가 제한적일 수 있어 모델 선택이 중요하다. 따라서 지금과 같은 점수는 어떤 모델을 사용하냐에 따라 달라질 수 있다.
![image](https://github.com/user-attachments/assets/9327ae2d-fb31-4199-99e9-2df484de82a3)

---

추가적으로 noise 값의 TARGET 값을 예측하지 않고 제거한 다음 LightGBM으로 예측했을 때가 가장 좋은 성능을 보여줬다. 결과는 아래와 같다.
```
Train Data Evaluation:
오차 행렬
[[5830 1506]
 [ 519 1692]]
정확도: 0.7879, 정밀도: 0.5291, 재현율: 0.7653,    F1: 0.6256, AUC:0.8581

Validation Data Evaluation:
오차 행렬
[[1482  362]
 [ 150  393]]
정확도: 0.7855, 정밀도: 0.5205, 재현율: 0.7238,    F1: 0.6055, AUC:0.8385
```
train data의 결과가 재현율이 높은데 정밀도가 낮다. 즉, 양성 클래스를 잘 잡아내지만, 많은 잘못된 긍정 (False Positive)도 예측하고 있다. 그래도 이전에 비해 과적합이 많이 해소된 것을 확인할 수 있다. 재현율, 정밀도에 대한 해결 방법으로 정밀도와 재현율의 균형 조정, 모델 복잡도 줄이기, 모델 앙상블이 있다.

결과를 kaggle에 제출했을 때 private, public 모두 이전과 많이 좋아진 것을 확인할 수 있다.
![image](https://github.com/user-attachments/assets/b524df6f-6b21-4745-8293-3dc522473c30)

베스트 파라미터는 아래와 같다.
```
Best params:  {'num_leaves': 44, 'min_child_samples': 30, 'min_child_weight': 7.026190601165056, 'max_depth': 3, 'subsample': 0.6587318096159958, 'colsample_bytree': 0.6087483644642271, 'learning_rate': 0.01961668163863785, 'scale_pos_weight': 2.762646582387838, 'reg_alpha': 2.1002708478959153, 'reg_lambda': 18.205433395806338, 'n_estimators': 586}
```

### 결론
이번 Santander Customer Satisfaction 프로젝트를 통해 다양한 데이터 분석 및 머신러닝 기법을 학습하고 적용할 수 있었습니다. 특히, 대규모의 익명화된 특성들을 다루며, 이 데이터셋의 특징을 파악하고 처리하는 데 상당한 노력을 기울였습니다. Feature Engineering을 통해 데이터를 정제하고, 이상치 및 노이즈를 처리하는 과정에서 여러 가지 접근 방식을 시도하였습니다.

모델 학습에서는 voting을 사용하기도 했지만 XGBoost, LightGBM, CatBoost와 같은 부스팅 기법을 사용해 높은 성능을 목표로 했습니다. 특히, 이 프로젝트에서 중요한 평가지표는 AUC (ROC 곡선 아래 면적)였으며, 이는 고객 불만족 예측이라는 문제 특성에 맞추어 재현율과 정밀도를 균형 있게 고려한 모델 평가를 가능하게 해주었습니다.

#### 한계점
프로젝트 진행 중 가장 아쉬웠던 점은 데이터의 비대칭성(imbalance)이었습니다. 데이터셋에서 불만족 고객의 비율이 매우 낮았기 때문에, 불균형 문제를 해결하기 위해 언더샘플링을 사용했지만, 데이터 손실로 인해 모델의 성능이 완벽히 개선되지는 않았습니다. 또한, 익명화된 특성 때문에 변수들의 의미를 명확히 이해하지 못하고, 그로 인해 효과적인 도메인 지식 기반의 피처 엔지니어링이 어려웠습니다.

#### 배운 점
이번 프로젝트는 데이터 전처리 및 특성 공학의 중요성을 다시 한번 깨닫게 해주었습니다. 특히 이상치 탐지, 노이즈 처리는 모델 성능을 크게 향상시키는 중요한 단계임을 확인할 수 있었습니다. 또한, 다양한 앙상블 기법과 그들의 장단점을 비교하는 경험을 통해 Voting, Boosting의 차이를 체감하며 Ensemble에 대해 더 깊게 공부할 수 있었습니다.

비록 목표했던 최고 성능에는 미치지 못했지만, 여러 시행착오를 거치며 성장할 수 있었던 의미 있는 프로젝트였습니다.

