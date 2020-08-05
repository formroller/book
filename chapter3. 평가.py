# CHAPTER 3. 평가
 머신러닝 프로세스
  - 가공/변환
  - 모델 학습/예측
  - 평가
  
[분류의 성능 평가 지표]
* 이진 분류 :  분류는 클래스 값 종류의 유형에 따라 긍정/부정과 같은 2개의 결괏값만 갖는다. 
* 멀티 분류 : 여러 개의 결정 클래스 값을 갖는다.
 
- 정확도 (Accuracy)
 - 오차행렬 (Confusion Matrix)
 - 정밀도 (Precision)
 - 재현율 (Recall)
 - F1 스코어
 - ROC AUC
=> 위 지표들은 이진/멀티 모두에 적용되지만, 이진 분류에서 더욱 중요하게 강조하는 지표이다
 
 # 01) 정확도
 - 직관적으로 모델 예측 성능을 나타내는 평가 지표이다.
   
 단점)
* 이진 분류의 경우, 데이터 구성에 따라 ML 모델의 성능을 왜곡할 수 있다.
* 불균형한 레이블 값 분포에서 ml 모델의 성능을 판단할 경우, 적합한 평가 지표가 아니다.

정확도(Accuracy) = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수

(MNIST 데이터 셋을 변환해 불균형한 데이터 셋으로 변환해 정확도 지표로 사용시 문제점)
MNIST : 0-9까지 숫자 이미지의 픽셀 정보를 갖으며 이를 통해 digit 예측


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metric import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    
    # 입력값으로 들어오는 X 데이터 셋의 크기만큼 모두 0값으로 만들어 변환
    def predict(self, X):
        return np.zeros( (len(X), 1), dtype = bool)
    
#사이킷런의 내장 데이터 셋인 load_digits()를 이용해 MINST 데이터 로딩
digits = load_digits()
#diabetes = load_diabetes()

#digits 번호가 7이면 True고, 이를 astype(int)로 1로 변환, 7번이 아니면 False이고 0으로 변환.
y = (digits.target==7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state = 11)

#불균형한 레이블 데이터 분포도 확인
print(f'레이블 테스트 세트 크기 : {y_test.shape}')
print('테스트 셋 레이블 0과 1의 분포도')
print(pd.Series(y_test).value_counts())

# Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train,y_train)
fakepred = fakeclf.predict(X_test)
print('모든 에측을 0으로 해도 정확도는 : {:.3f}'.format(accuracy_score(y_test, fakepred)))

# 02) 오차행렬 (confusion matrix, 혼동행렬)
 학습된 분류 모델이 예측을 수행하며 얼마나 헷갈리고(confusion) 있는지를 나타내는 지표
Negative / Positive로 분류

 예측 클래스)               Negative(0)               Positive(1)

 실제 클래스) Negative(0)   TN(True Negative)       FP(False Positive)
             
             Positive(1)   FN(False Negative)      TP(True Positive)
             
TN, TP, FN, FP 의미
앞 문자 T/F는 예측값과 실제 값이 같은가/틀린가를 나타낸다
뒤 문자 N/P는 예측 결괏값의 부정(0)/긍정(1)을 나타낸다.

TN : 예측값을 Negative 값 0으로 예측, 실제값 역시 Negative 값 0
     부정 예측 성공, 비환자로 예측해 실제 비환자임을 맞춤
     (오답이라고 맞췄다!!)
TP : 예측값을 Positive 값 1로 예측, 실제값 역시 Positive 값 1
     긍정 예측 성공 즉, 환자로 예측해 실제 환자임을 맞춤
     (맞췄다!!)

FN : 예측값을 Negative 값 0으로 예측, 실제값 역시 Positive 값 1
     부정 예측 실패, 비환자로 예측했으나 실제 환자
     (못 맞췄다..!)
FP : 예측값을 Positive 값 1로 예측, 실제값은 Negative 값 0
     긍정 예측 실패, 환자로 예측 했으나 비환자
     (잘못맞췄다..!)
     
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, fakepred)
TN : 405  FP : 0
FN : 45   TP : 0

정확도 = (TN + TP)/(TN+FP+FN+TP)


 # 03) 정밀도와 재현율
 - 불균형한 데이터 셋에서 정확도보다 선호되는 평가 지표

정밀도 = TP/(FP+TP)
재현율 = TP/(FN+TP)

정밀도 : 예측을 P로한 대상중 예측과 실제 값이 P로 일치한 데이터의 비율
        분모인 (FP+TP)는 예측을 P로 한 모든 데이터 건수
        분자인 TP는 예측과 실제 값이 P로 일치한 데이터 건수

재현율 : 실제 값이 P인 대상 중 에측과 실제 값이 P로 일치한 데이터의 비율
        분모인 (FN+TP)는 실제 값이 P인 모든 데이터 건수
        분자인 TP는 예측과 실제 값이 P로 일치한 데이터 건수
        
* 재현율이 상대적으로 더 중요한 지표인 경우 : 실제 Positive 양성인 데이터 예측을 Negative로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우
* 정밀도가 상대적으로 더 중요한 지표인 경우 : 실제 Negative 음성인 데이터 예측을 Positive로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우
=> 재현율과 정밀도 모두 TP를 높이는 데 동일하게 초점을 맞추지만,
   재현율은 FN(실제 Positive, 예측 Negative)를 낮추는데
   정밀도는 FP를 낮추는 데 초점을 맞춘다.
(사이킷런은 정밀도 계산을 위해 precision_socre(), 재현율 계산을 위해 recall_socre()를 제공)
from sklearn.metrics imort accuracy_score, precision_socre, recall_score, confusion_matrix

#정밀도/재현율 트레이드오프
정밀도 또는 재현율이 특별히 강조돼야 할 경우 분류의 결정 임곗값(Threshold)을 조정해 정밀도 or 재현율 수치를 높일 수 있다.
하지만 정밀도와 재현울은 상호 보완적인 평가 지표이므로 어느 한쪽을 강제로 높이면 다른 하나의 수치는 떨어지기 쉽다(정밀도/재현율의 트레이오프, trade-off)

predict_proba()
- 개별 데이터별로 예측 확률을 반환
- 학습이 완료된 사이킷런 Classifier 객체에서 호출이 가능하며 테스트 피처 데이 셋을 파라미터로 입력하면 테스트 피처 레코드의 개별 클래스 예측 확률을 반환한다.
- 예측 확률 결과 출력

Binarizer 클래스 사용법)
 threshold 변수를 특정 값으로 설정하고 Binarizer 클래스를 객체로 생성한다.
 생성된 Binarizer 객체의 fit_transform() 메서드를 이용해 넘파이 ndarray를 입력하면,
 입력된 ndarray의 값을 지정된 threshold보다 같거나 작으면 0값으로, 크면 1값으로 변환해 반환한다.
(예제)
from sklearn.preprocessing import Binarizer

X = [[1,-1,2],
     [2,0,0],
     [0,1.1,1.2]]

# X의 개별 원소들이 threshold보다 같거나 작으면 0을, 크면 1을 반환
binarizer = Binarizer(threshold = 1.1)
print(binarizer.fit_transform(X))
-> 입력된 X 데이터 셋의 Binarizer의 threshold값이 1.1 보다 크면 1, 작으면 0을 반환

from sklearn.metrics import precision_recall_curve

 # 04) F1 스코어
- 정밀도와 재현율을 결합한 지표
- 정밀도와 재현율이 어느 한 쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 갖는다.

F1 = 2/(1/recall + 1/precision) = 2 * (precision * recall) / (precision + recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)

 # 05) ROC 곡선과 AUC
ROC 곡선과 이에 기반한 AUC 스코어는 이진 분류의 예측 성능 측정에서 주요한 지표

* ROC 곡선 (Receiver Operation Characteristic Curve, 수신자 판단 곡선)
FPR(False Positive Rata)이 변할 때 TPR(True Positive Rate, 재현율)이 어떻게 변하는지 나타내는 곡선
FPR을 x축으로, TPR은 y축으로 잡으면 FPR변화에 따른 TPR의 변화가 곡선 형태로 나타난다.

TPR은 True Positive Rate의 약자로, 이는 재현율(=민감도)을 나타낸다.
따라서 TPR은 TP/(FN+TP)이다.
 또한 민감도에 대응하는 지표로 TNR(True Negative Rate)이라 불리는 특이성(Specificity)이 있다.
 
* 민감도(TPR), 실제값 P(양성)가 정확히 예측돼야 하는 수준을 나타낸다
              (질병이 있는 사람은 질병이 있는 것으로 양성 판정)
* 특이성(TNR), 실제값 N(음성)이 정확히 예측돼야 하는 수준을 나타낸다
              (질병이 없는 건강한 사람은 질병이 없는 것으로 음성 판정)
              
TNR인 특이성은 다음과 같이 구할 수 있다.
 TNR = TN / (FP + TN)

ROC 곡선의 X축 기준인 FPR(False Positive Rate)
 FPR = FP / (FP + TN) = 1 - TNR = 1 - 특이성

이므로 1 - TNR or 1 - 특이성으로 표현된다.

roc_curve()
(입력 파라미터)
y_true : 실제 클래스 값 array (array shape = [데이터 건수])
y_socre : predict_prob()의 반환값 array에서 Positive 컬럼의 예측 확률이 보통 사용된다. array,shape = [n_samples]
(반환 값)
fpr : fpr 값을 array로 반환
tpr : tpr 값을 array로 반환
threshold : threshold 값 array

=> 일반적으로 ROC 곡선 자체는 FPR과 TPR의 변환 값을 보는데 이용,
   분류의 성능지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC값으로 결정(1에 가까울 수록 좋다.)
from sklearn.metrics import roc_auc_score



