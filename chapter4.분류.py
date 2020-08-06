# CHAPTER 4. 분류
 # 01) 분류(Classification)의 개요
 - 지도학습 : 레이블(Label), 즉 명시적인 정답이 있는 데이터가 주어진 상태에서 학습하는 머신러닝 방식
 => 기존 데이터가 어떤 레이블에 속하는지 패턴을 알고리즘으로 인지한 뒤 새롭게 관측된 데이터에 대한 레이블을 판별하는 것.
 
 (분류는 다양한 머신러닝 알고리즘으로 구현할 수 있다.)
 * 베이즈 통계와 생성 모델에 기반한 나이브 베이즈
 * 독립변수와 종속변수의 선형 관계성에 기반한 로지스틱 회귀
 * 데이터 균일도에 따른 규칙 기반의 결정 트리
 * 개별 틀래스 간의 최대 분류 마진을 효과적으로 찾아주는 서포트 벡터 머신
 * 근접 거리를 기준으로 하는 최소 근접 알고리즘
 * 심층 연결 기반의 신경망
 * 서로 다른(또는 같은) 머신러닝 알고리즘을 결합한 앙상블
 
 (앙상블 )
 => 배깅 / 부스팅
 배깅 : 랜덤포레스트 
 
 부스팅 : 그래디언트 부스팅 (시간이 오래 걸리는 단점으로 최적화 모델 튜닝 어려움)
        -> XgBoost(Extra Gradient Boost)와 LightGBM 등장
           (기존 그래디언트 부스팅 예측 성능을 발전 시키며 시간 단축)
 
 (결정트리)
 - 쉽고 유연하게 적용될 수 있는 알고리즘.
 - 데이터 스케일링이나 정규화 등의 사전 가공의 영향이 매우 적다.
 단점) 
 => 예측 성능을 향상시키기 위해 복잡한 규칙 구조를 가져야 하며, 이로 인한 과적합이 발생해 예측 성능 저하될 수 있다.
 
 * 앙상블은 매우 많은 여러개의 약한 학습기를 결합해 확률적 보완과 오류가 발생한 부분에 대한 가중치를 계속 업데이트하며 예측 성능을 향상시키는데,
  결정 트리가 좋은 약한 학습기가 될 수 있기 때문이다

** 약한 학습기 : 예측 성능이 상대적으로  떨어지는 학습 알고리즘
 
 # 02) 결정 트리
 - ML알고리즘 중 직관적으로 이해하기 쉬운 알고리즘
 - 데이터데 있는 규칙을 학습해 트리 기반의 분류 규칙 생성.
 
 구성) 
 규칙 노드 : 규칙 조건이 되는 노드
 리프 노드 : 결정된 클래스 값
 서브 트리 : 새로운 규칙 조건마다 규칙 노드 기반의 서브트리가 생성된다.
  
단점) 
 - 많은 규칙이 있다 
 => 분류를 결정하는 방식이 복잡 => 과적합 발생.
 따라서, 트리의 깊이가 깊어질수록 결정 트리의 예측 성능이 저하될 가능성이 높다.
 (가능한 적은 결정 노드로 높은 예측 정확도를 가지려면 결정 노드의 규칙이 최대한 많은 데이터를 분류할 수 있어야 한다.)
즉, 과적합으로 정확도가 떨어진다. 

 
 # 정보의 균일도 측정 방법
 - 정보 이득 
* 엔트로피 개념 기반
* 1에서 엔트로피 지수를 뺀 값.
* 정보 이득이 높은 속성을 기준으로 분할한다.

엔트로피 : 데이터 집합의 혼잡도를 의미
          서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞이면 엔트로피가 낮다.
          
 지니계수
 - 불평등 지수 나타내낼 때 사용하는 계수
 - 0이 가장 평등, 1로 갈수록 불평등하다.
 - 다양성이 낮을 수록 균일도가 높다는 의몰 1로 갈수록 균일도가 높다
   (데이터가 다양한 값을 가질수록 평등, 특정 값을 가질수록 쏠릴 경우 불평등.)

=> 정보 이득이나 지니 계수가 높은 조건을 찾아 반복 분할 실시.

특징) 
'균일도' 룰을 기반으로해 알고리즘이 쉽고 직관적이다.
(정보의 균일도만 신경 쓰면 되므로 특별한 경우 제외하고 피처의 스케일과 정규화 같은 전처리 작업이 필요 없다.)

# 파라미터
- 사이킷런의 결정 트리 구현은 CART 알고리즘 기반.
(DecisionTreeClassifier와 DecisionTreeRegressor 클래스를 제공(CART, Classification And Regressor Trees))
(CART는 분류와 회귀에서 모두 사용할 수 있는 트리 알고리즘)
-> 모두 동일한 파라미터 사용

 1) min_samples_split
- 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는데 사용**
- 디폴트 : 2, 작게 설정할수록 분할되는 노드가 많아져 과적합 가능성 증가
- 과적합을 제어. 

 2) min_samples_leaf
- 말단 노드(Leaf)가 되기 위한 최소한의 샘플 데이터 수**
- Min_samples_split과 유사하게 과적합 제어 용도. 
  그러나 비대칭적 데이터의 경우 특정 클래스의 데이터가 극도록 작을 수 있으므로 이 경우 작게 설정 필요

 3) max_features
- 최적의 분할을 위해 고려할 최대 피처 개수. (디폴트는 None, 데이터 세트의 모든 피처를 사용해 분할 수행)**
- int형으로 지정하면 대상 피처의 개수, float형으로 지정하면 전체 피처 중 대상 피처의 퍼센트
- 'sqrt'는 전체 피처 중 sqrt(전체 피처 개수), 즉 sqrt(전체피처개수)만큼 선정
- 'auto'로 지정하면 sqrt와 동일
- 'log'는 전체 피터 중 log2(전체피처개수) 선정
- 'None'은 전체 피처 선정

 4) max_depth
- 트리의 최대 깊이 규정**
- 디폴트 None
- 깊이가 깊어지면 min_sample_split  설정대로 최대 분할해 과적합할 수 있으므로 적절한 값으로 제어 필요

 5) max_leaf_nodes
- 말단 노드(Leaf)의 최대 개수

# 결정 트리 모델의 시각화
Graphviz 패키지 사용 
- pip install graphviz
- conda install graphviz

[붓꽃 데이터 결정 트리 적용시 구현되는 서브트리 시각화]
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1) DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state = 151)

#2) 붓꽃 데이터 로딩, 학습/테스트 셋 분리
iris = load_iris()
X_train, X_test, y_train, y_test = trina_test_split(iris.data, iris.target, test_size = 0.2, random_state = 11)

#3) DecisionTreeClassifier 학습
dt_clf.fit(X_train, y_train)

#4) 시각화
from sklearn.tree import export_graphviz
# export_graphviz() 호출 결과로 out_file로 지정된 tree.dot 파일 생성
export_graphviz(dt_clf, out_file = 'tree.dot', class_names = iris.target_names, feature_names = iris.feature_names, impurity = True, filled = True)

import graphviz
# 위에서 생성된 tree.dot 파일을 Graphviz가 읽어 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

* gini : value = []로 주어진 데이터 분포에서의 지니 계수
* samples : 현 규칙에 해당하는 데이터 건수
* value : 클래스 값 기반 데이터 건 수

import seaborn as sns
import numpy as np

# feature instance 추출
print(f' Feature importance : \n {np.round(dt_clf.feature_importances_,3)}')

# feature별 improtance 매핑
for name, value in zip(iris.feature_names, dt_clf.feature_importances_):
    print('{0} : {1:3f}'.format(name, value))

# feature importance를 column 별로 시각화하기
sns.barplot(x = dt_clf.feature_importances_, y = iris.feature_names)

# 결정 트리 과적합
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

plt.title('3 Class values with 2 Feature Sample data creation')

# 2차원 시각화를 위해 피처는 2개, 클래스는 3가지 유형의 분류 샘플 데이터 생성
X_features, y_labels = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, n_classes = 3, n_clusters_per_class = 1, random_state = 1)

# 그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색으로 표시된다.
plt.scatter(X_features[:,0], X_features[:,1], marker = 'o', c = y_labels, s=25, edgecolor = 'k')


# visualize_boundary : 모델이 클래스 값을 예측하는결정 기준을 색상과 결계로 나타내는 함수
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    
    # 학습 데이터 scatter plot으로 나타내기
    ax.scatter(X[:,0], X[:,1], c=y, s=25, cmap='rainbow', edgecolor='k', clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start, xlim_end = ax.get_xlim()
    ylim_start, ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이터로 model 학습.
    model.fit(X,y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200), np.linspace(ylim_start, ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    #contourf()를 이용해 class boundary를 시각화 수행.
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha = 0.3, levels=np.arange(n_classes + 1) - 0.5, cmap = 'rainbow', clim=(y.min(), y.max()),zorder = 1)
    

 # 특정한 트리 생성 제약 없는 결정 트리의 학습과 결정 경계 시각화
from sklearn.tree import DecisionTreeClassifier    
dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels) 

# 결정 트리 실습 - 사용자 행동 인식 데이터 세트
(UCI 머신러닝 리포지토리 제공하는 사용자 행동 인식 데이터 세트 예측 분류)
http://archive.ics.uci.edu/ml/machine-learning-databases/00240/

import os
os.getcwd()
os.chdir('~\Users\yongjun/.spyer-py3/human_act/__MACOSX')
os.path.abspath('Users\yongjun\.spyder-py3')

import pandas as pd
import matplotlib.pyplot as plt

# features.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음. 이를 DataFrame로 로드.
feature_name_df = pd.read_csv('features.txt', sep='\s+', header=None, names=['columns_index','columns_name'])

# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:,1].values.tolist()
print(f'전체 피처명에서 10개만 추출 : {feature_name[:10]}')

=> 피처명은 인체의 움직임과 관련된 속성/표준편타가 X,Y,Z축 값으로 돼 있음을 유추할 수 있다.

#DataFrame 생성하는 로직을 갖춘 함수
import pandas as pd

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data = old_feature_name_df.groupby('columns_name').cumcount(),columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how = 'outer')
    new_feature_name_df = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df['column_name'] = new_feature_name_df.drop(['index'], axis = 1)
    return new_feature_name_df

def get_human_datasets():
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep로 할당
    feature_name_df = pd.read_csv('features.txt', sep = '\s+', header = None, names = ['column_index','columns_name'])
    
    # 중복된 feature명을 새롭게 수정하는 get_new_feature_name_df()를 이용해 새로운 feature명 DataFraem생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = feature_name_df.iloc[:,1].values.tolist()
    
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    # 학습 피처 데이터 세트와 테스트 피처 데이터를 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('./train/X_train.txt', sep = '\s+', names=feature_name)
    X_test = pd.read_csv('./test/X_test.txt', sep = '\s+', names=feature_name)
    
    
    # 학습 레이블과 테스트 레이블 데이터를 DataFrame을 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('./train/y_train.txt', sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('./train/y_test.txt', sep='\s+', header=None, names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_datasets()



 # 03) 앙상블 학습
 # 04) 랜덤 포레스트
 # 05) GBM
 # 06) XGBoost
 # 07) LightGBM
 # 08) 분류 실습 - 캐글 산탄데르 고객 만족 예측
 # 09) 분류 실습 - 캐글 신용카드 사기 검출
 # 10) 스태킹 앙상블