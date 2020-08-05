 CHAPTER 3-2) 피마 인디언 당뇨병 예측
 # 06) 피마 인디언 당뇨병 예측

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

os.getcwd()
os.chdir('./.spyder-py3/1.diabetes')

[features]
* Pregnancies : 임신 횟수
* Glucose : 포도당 부하 검사 수치
* BloodPressure : 혈압(mm Hg)
* SkinThickness : 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
* Insulin : 혈청 인슐린(mu U/ml)
* BMI : 체질량지수(체중(kg)/(키(cm))^2)
* DiabetsPedigreeFunction : 당뇨 내력 가중치 값
* Age : 나이
* Outcome : 클래스 결정 값(0 or 1)


diabetes_data = pd.read_csv('diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
diabetes_data.head(3)
# -> 전체 데이터 중 N:500, P:268

# feature type과 null 개수
diabetes_data.info()
-> Null 없음, 숫자 타입


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156, stratify = y)
trian_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 12)

#평가 결과 출력 함수
def get_clf_eval(y_test = None, pred = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    # ROC - AUC 추가
    roc_auc = roc_auc_score(y_test, pred)
    print('오차행렬')
    print(confusion)
    # ROC - AUC print 추가
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f},  F1 : {3:.4f}, AUC : {4:.4f}'
          .format(accuracy, precision, recall, f1, roc_auc))

#임계값별 정밀도와 재현율 그래프 출력 함수
def precision_recall_curve_plot(y_test = None, pred_proba_c1 = None):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray추출.
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    
    # X축을 threshold값으로, y축은 정밀도, 재현율 값으로 각각 Plot 수행, 정밀도는 점선 표시
    plt.figure(figsize = (8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle = '--', label = 'percision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall')
    
    # threshold 값 X 축의 sclae을 0,1 단위로 변경
    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # X축, ycnr label과 legend, grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
    
# 피처 데이터 세트 X, 레이블 데이터 세트 y 추출
# 마지막 컬럼인 Outcome이 레이블 값. (-1로 추출)
X = diabetes_data.iloc[:,:-1]
y = diabetes_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156, stratify = y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)

#전체 데이터의 65%가 Negative이므로 정확도보다 재현율 성능에 초점을 맞춘다.
#정밀도 재현율 곡선을 보고 임곗값별 정밀도와 재현율 값의 변화 확인 (precision_recall_curve_plot() 함수 이용)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
precision_recall_curve_plot(y_test, pred_proba_c1)
=>0.42 정도로 낮출 경우 정밀도와 재현율의 균형이 맞춰진다.(그러나 여전히 두 지표의 값이 낮다.)


# 데이터 확인
diabetes_data.describe()
plt.hist(diabetes_data['Glucose'], bins = 10) # 0값이 일정 수준 존재한다.

#[ 0값의 대비 전체 데이터 비율 확인]
# 0값을 검사할 피처 명 리스트
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes.columns
# 전체 데이터 건 수
total_count = diabetes_data['Glucose'].count()

# 피처별 반복하며 데이터 값이 0인 데이터 건수 추출하고, 퍼센트 계산
for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    print('{0} 0데이터 수는 {1}, 퍼센트는 {2:.2f}%'.format(feature, zero_count, 100*zero_count/total_count))
    
# zero_feature 리스트 내부에 저장된 개별 피처들에 대해 0값을 평균 값으로 대체
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0,mean_zero_features)

X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:,-1]

#StandardScaler 클래스를 이용해 피처 데이터 세트에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 156, stratify = y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)
=> 데이터 변환(0-> 평균값 대치)과 스케일링을 통해 성능 수치 개선 


# 임곗값 변환에 따른 성능 수치 변화 
from sklearn.preprocessing import Binarizer

def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds 리스트 객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict)
        
# 분류 결정 임곗값 변화에따른 재현율과 평가 성능 수치 개선 확인 
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.5]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1,1), thresholds)
=> 임곗값이 0.48일때 성능이 가장 우수

* predict() 메서드는 임곗값 임의 수정 불가
따라서 별도의 로직으로 이를 구해야 한다. 
Binarizer 클래스를 이용해 predice_proba()로 추출한 예측 결과 확률 값을 변환해 변경된 임곗값에 따른 예측 클래스 값 출력

# 임곗값을 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold = 0.48)

# 위에서 구한 lr_clf의 predict_proba() 예측 확률 array에서 1에 해당하는 컬럼값을 Binarizer 변환.
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1,1))

get_clf_eval(y_test, pred_th_048)
