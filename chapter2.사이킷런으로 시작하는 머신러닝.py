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
