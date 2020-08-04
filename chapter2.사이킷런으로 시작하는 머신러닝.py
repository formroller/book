# CHAPTER 2. 사이킷런으로 시작하는 머신러닝
# 01) 사이킷런 소개와 특징
(특징)
 - 파이썬 기반의 다른 머신러닝 패키지도 사이킷런 스타일의 API를 지향할 정도로 쉽고 가장 파이썬스러운 API를 제공
 - 머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API를 제공한다.
 - 오랜 기간 실전 환경에서 검증됐으며, 매우 많은 환경에서 사용되는 성숙한 라이브러리이다.
 
(설치)
 - 제거하지 않을 경우 별도의 설치 필요치 않다.
 - 제거되서 설치해야하는 경우
 conda install scikit-learn
 pip install scikit-learn
(버전확인)
import sklearn
print(sklearn.__version__)

# 02) 첫 번째 머신러닝 만들어 보기 - 붓꽃 품종 예측하기
붓꽃 데이터 세트로 붓꽃의 품종을 분류,
 - 분류 (Classification)
 ; 분류는 지도학습 방법의 하나.
 ; 지도학습은 학습을 위한 다양한 피처와 분류 결정값인 레이블(Label) 데이터로 모델 학습한 뒤
   별도의 테스트 데이터 세트에서 미지의 레이블을 예측하는 것.

 학습하기 위해 주어진 데이터(train set)
 평가를 위해 주어진 데이터 (test set)
 
sklearn.datasets : 사이킷런에서 자체적으로 제공하는 데이터 세트를 생성하는 모듈의 옴임
sklearn.tree : 트리 기반 ML 알고리즘을 구현한 클래스의 모임
sklearn.model_selection : 학습, 검증, 예측 데이터로 데이터를 분리하거나 최적의 하이퍼 파라미터로 평가하기 위한 모듈


[계획 : load_iris()를 이용해 ML 알고리즘 중 의사 결정 트리(Decision Tree)로, 이를 구현한 DexisionTreeClassifier를 적용한다.]

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 

import pandas as pd

# 붓꽃 데이터 세트 로딩
iris = load_iris()

# iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있다.
iris_data = iris.data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있다.
iris_label = iris.target
print(f'iris target값 : {iris_label}')
print(f'iris target명 : {iris.target_names}')

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환
iris_df = pd.DataFrame(data = iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head()

 - Seotosa : 0
 - Versicolor : 1
 - Virginica : 2

# data set 분할
X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size = 0.2, random_state = 11) 

 - X_train : 학습용 피처 data set
 - X_test : 테스트용 피처 data set
 - Y_train : 학습용 레이블 data set
 - y_test : 테스트용 레이블 data set
 
 - test_size, 전체 데이터 중 테스트 데이터 세트의 비율
 - random_state, 호출할 때마다 같은 학습 / 테스트 용 데이터 세트를 생성하기 위해 주어지는 난수 발생 값
  (= seed와 같은 의미로 숫자 자체 값은 상관 없다.)

# DecisionTreeClassifier 객체 생성
df_clf = DecisionTreeClassifier(random_state = 11)

# 학습 수행
df_clf.fit(X_train, Y_train)

# 학습 완료된 DTC 객체에서 테스트 데이터 세트로 예측 수행
pred = df_clf.predict(X_test)

# 예측 정확도 확인 : accuracy_socre()
from sklearn.metrics import accuracy_score
print('예측 정확도 : {0:.4f}'.format(accuracy_score(Y_test, pred)))

 [붓꽃 데이터 세트로 분류 예측한 프로세스]
 1. 데이터 세트 분리 : 데이터를 학습 / 테스트 데이터로 분리
 2. 모델 학습 : 학습 데이터를 기반으로 ML알고리즘 적용해 모델 학습
 3. 예측 수행 : 학습된 ML 모델을 이용해 데이트 데이터의 분류 예측
 4. 평가 : 예측된 결괏값과 테스트 데이터의 실제 결괏값을 비교해 ML모델 성능 비교
 
# 03) 사이킷런의 기반 프레임 워크 익히기
 # Estimator 이해 및 fit(), predict() 메서드
 
 .fit() : ML모델 학습
 .predict() : 학습된 모델의 예측
 
 (지도학습)
분류(Classification) -> classifier
 (classifier) 
 - DT
 - RF
 - GB
 - GaussianNB
 - SVC
 
회귀(Regression)     -> regressor
 (regressor)
 - LR
 - Ridge
 - Lasso
 - RF regressor
 - GB regressor
 => 위 두 알고리즘을 구현한 클래스를 Estimator라 한다.
 
# => cross_val_score()와 같은 evaluation 함수, GridSearchCV와 같은 하이퍼 파라미터 튜닝을 지원하는 클래스의 경우 Estimator를 인자로 받는다.

[사이킷런의 주요모듈]
#               sklearn.~
1) 예제 데이터  datasets, 사이킷런에 내장된 예제 제공하는 데이터 세트

2) 피처 처리    preprocessing, 전처리에 필요한 다양한 가공 기능 제공(인코딩, 정규화, 스케일링 등)
               feateure_selection, 알고리즘에 큰 영향을 미치는 피처를 우선순위별 선택 작업 수행
               feature_extraction, 텍스트나 이미지 데이터의 벡터화된 피처를 추출하는데 사용

3) 피처처리& 차원 축소 decomposition, 차원 축소와 관련한 알고리즘을 지원하는 모듈.(PCA, NMF, Truncated SVD등을 통해 차원 축소 수행)

4) 데이터 분리, 검증 & 파라미터 튜닝 model_selection, 교차 검증을 위한 학습/테스트용 분리, 그리드 서치(Grid Search)로 최적 파라미터 추출

5) 평가 metrics, 분류/회귀/클러스터링/페이와이즈(Pairwise)에 대한 다양한 성능 측정 방법 제공

6) ML 알고리즘 ensemble, 앙상블 알고리즘 제공
                        랜덤 포레스트, 에이다 부스트, 그래디언트 부스팅 등을 제공
               
               linear_model, 주로 선형 회귀,릿지(Ridge),라쏘(Lasso) 및 로지스틱 회귀 등 회귀 롼련 알고리즘 지원
                             또한 SGD(Stochastic Gradient Descent) 관련 알고리즘도 제공
 
               naive_bayes, 나이브 베이즈 알고리즘 제공. (가우시안NB, 다항 분포 NB등)
               
               neighbors, 최근접 이웃 알고리즘 제공(K-NN등)
               
               SVM, 서포트 벡터 머신 알고리즘 제공
               
               tree, 의사 결정 트리 알고리즘 제공
               
               cluster, 비지도 클러스터링 알고리즘 제공(K-평균, 계층형, DBSCAN등)
               
7) 유틸리티 pipline, 피처 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 함께 묶어 실행할 수 있는 유틸리티 제공

# 내장된 예제 데이터 세트
datasets.load_~
(분류)
breast_cancer(), 위스콘신 유방암 피처들과 악성/음성 레이블 
digits(), 0-9까지 숫자의 이미지 픽셀 
iris(), 붓꽃에 대한 피처

(회귀)
boston(), 미국 보스턴의 집 피처들과 가격에 대한 데이터 세트
diabetes(), 당뇨 데이터 세트

(분류와 클러스터링을 위한 표본 데이터 생성기)
datasets.make_classifications(), 분류를 위한 데이터 세트 생성. 
                                 특히 높은 상관도, 불필요한 속성 등의 노이즈 효과를 위한 데이터를 무작위로 생성

datasets.make_blobs(), 클러스터링을 위한 데이터 세트를 무작위로 생성한다. 군집 지정 개수에 따라 여러가지 클러스터링을 위한 데이터 세트 생성

data : 피처의 데이터 세트를 읨
target : 분류 시 레이블 값, 회귀 시 숫자 결괏값 데이터 세트
target_names : 개별 레이블의 이름
feature_names : 피처 이름
DESCR : 데이터 세트에 대한 설명과 피처의 설명 의미

data/target은 넘파이 배열(ndarray) 타입
target_names/feature_names는 넘파이 배열 또는 리스트 타입
DESCR은 스트링 타입

from sklearn.datasets import load_iris
iris_data = load_iris()
print(type(iris_data))

 - Bunch클래스는 딕셔너리 자료형과 유사.
keys = iris_data.keys()
print(f'붓꽃 데이터 세트의 키값 : {keys}')

(iris_data.feature_names)
print(f'\n feature_names의 type : {type(iris_data.feature_names)}')
print(f' feature_names의 shape : {len(iris_data.feature_names)}')
print(iris_data.feature_names)

(iris_data.target_names)
print(f'\n target_names의 type : {type(iris_data.target_names)}')
print(f' target_names의 shape : {len(iris_data.target_names)}')
print(iris_data.target_names)

...

# 04) Model Selection 모듈 소개
사이킷런의 model_selection 모듈은 학습 데이터와 테스트 세트를 분리하거나 교차 검증 분할 및 평가와 
Estimator의 하이퍼 파라미터를 튜닝하기 위한 다양한 함수와 클래스를 제공한다.

# 학습 / 테스트 데이터 세트 분리 - train_test_split()
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

# 학습 데이터 세트로 예측 수행
pred = dt_clf.predict(train_data)
print(f'예측 정확도 : {accuracy_score(train_label,pred)}')
=> 이미 학습한 데이터 기반으로 예측했기 때문에 정확도 100% 도출.
-> 따라서 예측을 수행하는 데이터 세트는 학습을 수행한 학습용 데이터 세트가 아닌 전용 테스트 데이터 세트여야 한다.

# train_test_split
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(iris_data.data, iris_data.target, test_size = 0.3, random_state = 121)

dt_clf.fit(train_x, train_y)
pred = dt_clf.predict(test_x)
print('예측 정확도 : {0:.4f}'.format(accuracy_score(test_y, pred)))

# 교차검증
데이터 편중을 막기위해 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가 수행
과적합 : 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우 예측 성능히 현저히 저하되는 경우

# K-폴드 교차 검증
- k개의 데이터 폴드 세트를 만들어 k번 만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법

(5 폴드 교차 검증, k=5)
- 5개의 폴드된 데이터 세트를 학습고 검증을 위한 데이터 세트로 변경하며 5번 평가를 수행한 뒤, 5개의 평가를 평균한 결과를 예측 성능을 평가한다.
1) 먼저 데이터 세트를 k등분(5등분)한다.
2) 첫 번째 반복에서 처음부터 4개 등분을 학습 세트, 마지맛 5번째 등분 하나를 검증 세트로 설정하고 학습 세트에서 학습 수행
3) 검증 데이터 세트에서 평가 수행
... 위와 같이 첫 번째 평가 수행하고 나면 두 번째 반복에서 다시 비슷한 하습과 평가 작업을 수행.
    단, 이번에는 학습 데이터와 검증 데이터를 변경한다
    이와 같이 학습 데이터 세트와 검증 데이터 세트를 점진적으로 변경하면서 마지막 5번째(K번째)까지 학습과 검증을 수행하는 것이 K 폴드 교차 검증
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state = 156)

# 5개 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits=5)
cv_accuracy = []
print(f'붓꽃 데이터 세트 크기 : {features.shape[0]}')
n_iter = 0

# KFold 객체의 split()를 호출하면 폴드 별 학습/검증용 테스트 로우 인덱스를 array로 변환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 인덱스를 이용해 학습/검증용 테스트 데이터 추출
    X_trian, X_test = features[train_index], features[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf.fit(X_train, Y_train)
    pred = dt_clf.predict(X_test)
    n_iter +=1
    # 반복마다 정확도 측정
    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'\n# {n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기 :{train_size}, 검증 데이터 크기 : {test_size}')
    print(f'#{n_iter} 검증 세트 인덱스:{test_index}')
    cv_accuracy.append(accuracy)
    
# 개별 iteration별 정확도를 합해 평균 정확도 계산
print(f'\n## 평균 검증 정확도 : {np.mean(cv_accuracy)}')

# Straified K 폴드
 - 불균형한 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K 폴드 방식이다.
 * 불균형한 분포도를 가진 레이블 데이터 집합은 특정 레이블 값이 특이하게 많거나 적어 값의 분포가 한쪽으로 치우치는 것을 말한다.
 
import pands as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()
=> Setosa 50, Versicolor 50, Virginica 50

# 이슈가 발생하는 현상을 도출하기 위해 3개의 폴드 세트를 KFold로 생성하고, 각 교차 검증 시마다 생성되는 학습/검증 레이블 데이터 값의 분포도 확인
kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter +=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'## 교차 검증 : {n_iter}')
    print(f'학습 레이블 데이터 분포 :\n {label_train.value_counts()}')
    print('검증 레이블 데이터 분포 :\n', label_test.value_counts())
    
# KFold와 차이점
- KFold로 분할된 레이블 데이터 세트가 전체 레이블 값의 분포도를 반영하지 못하는 문제를 해결한다.
- StraifiedKfold는 레이블 데이터 분포도에 따라 학습/검증 데이터를 나누기 때문에 split() 메서드에 인자로 피처 세트뿐 ㅇkslfk 데이블 데이터 세트도 반드시 필요하다
  (K 폴드의 경우 레이블 데이터 세트는 split() 메서드의 인자로 입력하지 않아도 무방하다.)

## (StraifiedKFold= 3)    
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 3)
n_iter = 0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter = 0
    lable_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'## 교차 검증 : {n_iter}')
    print(f'학습 레이블 데이터 분포 : \n {label_train.value_counts()}')
    print(f'검증 레이블 데이터 분포 : \n {label_test.value_counts()}')
    

## StratifiedKFold 검증
dt_clf = DecisionTreeClassifier(random_state = 151)

skfold = StratifiedKFold(n_splits=3)
n_iter = 0
sc_accuracy = []

# StratifiedKFold의 split() 호출시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index in skfold.split(features, label):
    # split()으로 반환된 인덱스를 이용해 학습용/검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    Y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf.fit(X_train, Y_train)
    pred = dt_clf.predict(X_test)
    
    # 반복마다 정확도 측정
    n_iter +=1
    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차검증 정확도 : {1}, 학습 데이터 크기 :{2}, 검증 데이터 크기 : {3}'.format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)
    
# 교차 검증별 정확도 및 평균 정확도 계산
print(f'\n## 교차 검증별 정확도 : {np.round(cv_accuracy,4)}')
print(f'## 평균 검증 정확도 : {np.mean(cv_accuracy)}')

# - 일반적으로 분류에서의 교차 검증은 Stratified K 폴드로 분할돼야 한다.
# - 회귀에서는 Stratified K 폴드가 지원되지 않는다.
=> 회귀의 결정값은 이산값 형태의 레이블이 아니라 연속된 숫자값이기 때문에 결정값별로 분포를 정하는 의미가 없기 때문.

# 교차 검증을 보다 간편하게 - cross_val_score()
 - 사이킷런에서 제공하는 교차검증 API
1) 폴드 세트를 설정
2) for 루프에서 반복으로 학습 및 테스트 데이터의 인덱스를 추출
3) 반복적으로 학습과 예측을 수행하고 예측 성능을 반환

cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None,
                pre_dispatch='2*n_jobs')
주요 파라미터)
 - estimator : 사이킷런의 분류 알고리즘인 Classifier 또는 회귀 알고리즘인 Regressor를 의미
 - X : 피처 데이터 세트
 - y : 레이블 데이터 세트
 - scoring : 예측 성능 평가 지표
 - cv : 교차 검증 폴드 수
 
 # corss_val_score() 사용법
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state = 156)

data = iris_data.data
lable = iris_data.target

# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(dt_clf, data, lable, scoring = 'accuracy', cv = 3)
print(f'교차 검증별 정확도 : {np.round(scores,4)}')
print(f'평균 검증 정확도 : {np.round(np.mean(scores),4)}')

# GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에
Grid_parameters = {'max_depth' : [1,2,3],
                   'min_samples_split' : [2,3]}

 - 교차 검증을 기반으로 이 하이퍼 파라미터의 최적값을 찾게 한다.
 - 여러 종류의 하이퍼 파라미터를 다양하게 테스트하며 최적의 파라미터를 편리하게 찾아준다.
 주요 파라미터) 
 - estimator : classifier, regressor, pipline
 - param_grid : Key + 리스트 값을 갖는 딕셔너리가 주어진다.
                Estimator의 튜닝을 위해 파라미터명과 사용될 여러 파라미터 값을 지정한다.
 - scoring : 예측 성능을 측정할 평가 방법을 지정.
             사이킷런의 성능 평가 지표를 지정하는 문자열(예:정확도 - accuracy)로 지정하거나 별도의 성능 평가 지표 함수도 지정할 수 있다.
 - cv : 교차 검증을 위해 분할되는 학습/테스트 세트의 개수 지정
 - refit : (디폴트가 Trued) True로 생성 시 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킨다.
 
 # GridSearchCV API 실습
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 데이터 로디하고 학습 데이터와 테스트 데이터 분리
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 121)
dtree = DecisionTreeClassifier()

## 파라미터를 딕셔너리 형태로 설정
params = {'max_depth' : [1,2,3], 'min_samples_splits':[2,3]}

import pandas as pd

# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트 수행 설정
### refit=True가 default임. True면 가장 좋은 파라미터 설정으로 재학습 시킴
grid_dtree = GridSearchCV(dtree, param_grid = params, cv = 3, refit = True)
# 붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_) ## sklearn 0.22.1 -> cv_results_ 사용 불가
 - params 컬럼에는 수행할 때마다 적용된 개별 하이퍼 파라미터값을 나타낸다
 - rank_test_score는 하이퍼 파라미터별로 성능이 좋은 score 순위를 나타낸다.
   (이때, 1이 가장 뛰어난 순위이며 이때의 파라미터가 최적의 파라미터 이다)
 - mean_test_scroe는 개별 하이퍼 파라미터별 CV의 폴딩 테스트 세트에 대해 총 수행한 평가 평균값.
 - GridSearchCV 객체의 fit()을 수행하면 최고 성능을 나타낸 하이퍼 파라미터 값과 그때의 평가 결과 값이 각각
  best_params_, best_score_ 속성에 기록된다.
 
# 05) 데이터 전처리
 - 어떤 데이터를 입력으로 가지느냐에 따라 결과도 크게 달라질 수 있다.
   (Garbage In - Garbage Out)
 - ML알고리즘은 결측치를 허용하지 않는다.
 
 # 데이터 인코딩
 모든 문자열 값은 인코딩돼서 숫자 형으로 변환해야 한다.
 문자열 피처는 일반적으로 카테고리형/텍스트형의 피처를 의미한다.
 
 - 레이블 인코딩
 - 원-핫 인코딩
 
 1) 레이블 인코딩(LabelEncoder)
 카테고리 피처를 코드형 숫자 값으로 변환하는 것.
 (ex-상품 구분)  TV:1, 냉장고:2, 전자레인지:3, 컴퓨터:4, 선풍기:5
 
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

#LabelEncoder를 객체로 생성한 후, fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

print(f'인코딩 변환값 : {labels}')  # 인코딩 변환
print(f'인코딩 클래스 : {encoder.classes_}') # 속성값 확인
print(f'디코딩 원본 값 : {encoder.inverse_transform([4,5,2,0,1,1,3,3])}') # 인코딩된 값을 다시 디코딩

 2-1) 원-핫 인코딩(one-hot encoding)
고유 값에 해당하는 컬럼에만 1을 표시, 나머지는 0.
여러 개의 속성 중 단 한 개의 속성만 1로 표시

(주의점)
1.OneHotEncoder로 변환하기 전에 모든 문자열 값이 숫자형 값으로 변환돼야 한다.
2.2차원 데이터가 필요하다.

ex)
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items =['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '믹서', '믹서']

#먼저 숫자 값으로 변환을 위해 LabelEncoder로 변환
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
#2차원 데이터료 변환
labels = labels.reshape(-1,1)

#원-핫 인코딩 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원-핫 인코딩 데이터 ')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)

 2-2)원-핫 인코딩 (get_dummies)
OneHotEncoder와 다르게 문자열 카테고리 값을 숫자 형으로 변환할 필요 없이 바로 변환 가능

import pandas as pd
df = pd.DataFrame({'item':['tv','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)

 # 피처 스케일링 정규화
- standardization (표준화)
  평균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환하는 것을 의미.
  
- normalization (정규화)
  서로 다른 피처의 크기를 통일하기 위해 크기를 변화해 주는 개념.
  
 # StandardScaler
 - 표준화를 쉽게 지원하기 위한 클래스
 -> 개별 피처를 평균이 0이고 분산이 1인 값으로 변환
 
 (예시)
from sklearn.datasets import load_iris
import pandas as pd
# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환한다.
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data = iris_data, columns=iris.feature_names)

(standardScaler 변환 전)
print('feature들의 평균값')
print(iris_df.mean())
print('\nfeature 들의 분산 값')
print(iris_df.var())

(standardScaler 변환)
from sklearn.preprocessing import StandardScaler

#StandardScaler객체 생성
sclaer = StandardScaler()
#StandardSclaer로 데이터 세트 변환. fit()과 transfrom() 호출.
sclaer.fit(iris_df)
iris_scaled = sclaer.transform(iris_df)

#transform() 시 스케일 변환된 데이터 세트가 Numpy ndarrat로 반환돼 이를 DataFrame로 변환
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print('feature들의 평균값')
print(iris_df_scaled.mean())
print('\nfeature들의 분산값')
print(iris_df_scaled.var())
=> 모든 컬럼 값의 평균이 0에 가까운 값으로, 분산은 1에 가까운 값으로 변환

 # MinMaxSclaer
 - 데이터 값을 0과 1사이의 범위 값으로 변환한다.
   (음수가 있을 경우 -1 ~ 1로 변환)
 - 데이터 분포가 가우시안 분포가 아닐 경우 Min,Max Scale을 적용해 볼 수 있다.
(예제)
from sklearn.preprocessing import MinMaxScaler

#MinMaxSclaer 객체 생성
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_sclaed = scaler.transform(iris_df)

#transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 반횐대 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print('feature들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature들의 최댓값')
print(iris_df_scaled.max())

# 129P 06) 사이킷런으로 수행하는 타이타닉 생존자 예측
