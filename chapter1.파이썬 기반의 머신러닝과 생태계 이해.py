# CHAPTER 01. 파이썬 기반의 머신 러닝과 생태계 이해
 # 01) 머신러닝의 개념
 애플리케이션을 수정하지 않고도 데이터를 기반으로 패턴을 학습학고 결과를 예측하는 알고리즘 기법
 단점) 데이터에 매우 의존적이다. (Garbage In - Garbage Out : 좋은 품질의 데이터를 갖추지 못한다면 머신러닝 수행 결과도 좋을 수 없다.)
 
 [머신러닝 분류]
 
 (지도학습)
  - 분류
  - 회귀
  - 추천 시스템 
  - 시각/음성 감지/인지
  - 텍스트 분석, nLP
  
  (비지도 학습)
  - 클러스터링
  - 차원 축소
  - 강화학습
  
(파이썬과 R 기반의 머신러인 비교 분석 관점 비교)
R : 통계 전용 프로그램 언어
파이썬 : 직관적 문법과 객체지향, 함수형 프로그래밍 모두를 포괄하는 유연한 프로그램 아키텍처, 다양한 라이브러리 등의 강점을 갖는 언어

# 02) 파이썬 머신러닝 생태계를 구성하는 주요 패키지
 # 파이썬 기반의 머신러닝을 익히기 위해 필요한 패키지

 - 머신러닝 패키지 : 사이킷런(Scikit-Learn)
 - 행렬/선형개수/통계 패키지 : 넘파이(Numpy, 행렬과 선형대수), 사이파이(SciPy, 자연과학과 통계)
 - 데이터 핸들링 : 판다스(2차원 데이터 처리 패키지, Matplotlib을 호출해 시각화 지원)
 - 시각화 : seaborn, Matplotlib(세분화된 API로 익히기 쉽지않고 투박)


[Jupyter Notebook install]
Anaconda prompt...
(conda)
conda install -c -conda-forge notebook
(pip)
pip install notebook
jupyter notebok

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 03) 넘파이(Numpy)
Numpy : Numerical Python 

# [넘파이 ndarray 개요]
ndarray : 넘파이의 기본 데이터 타입
          다차원(Multi-dimension) 배열을 쉽게 생성하고 연산 가능
          
array1 = np.array([1,2,3])
print(f'array1 type : {type(array1)}')
print(f'array1 array 형태 : {array1.shape}')
# 1차원 array로 3개의 데이터 포함

array2 = np.array([[1,2,3],
                   [4,5,6]])
print(f'array2 type : {type(array2)}')
print(f'array2 array 형태 : {array2.shape}')

array3 = np.array([[1,2,3]])
print(f'array3 type : {type(array3)}')
print(f'array3 array 형태 : {array3.shape}')
# 2차원 array로 1로우와 3컬럼 데이터

print(f'array1 : {array1.ndim}차원, array2 : {array2.ndim}차원, array3 : {array3.ndim}차원')
array() 함수의 인자로는 파이썬의 리스트 객체가 주로 사용된다.
리스트 []는 1차원, 리스트 [[]]는 2차원 형태.
ndarray내의 데이터는 같은 타입만 허용.
데이터 타입은 dtype로 확인 가능.

# dtpye , 데이터 타입 확인
# (같은 데이터 타입)
li1 = [1,2,3]
print(type(li1))
arr1 = np.array(li1)
print(type(arr1))
print(arr1, arr1.dtype)

# (다른 데이터 타입)
li2 = [1,2,'test']
arr2 = np.array(li2)
print(arr2, arr2.dtype)
li3 = [1,2,3.0]
arr3 = np.array(li3)
print(arr3, arr3.dtype)

# astype , 데이터 타입 변경
arr_int = np.array([1,2,3])
arr_float = arr_int.astype('float64')
print(arr_float, arr_float.dtype)

arr_int1 = arr_float.astype('int32')
print(arr_int1, arr_int1.dtype)

arr_float1 = np.array([1.1, 2.1, 3.1])
arr_int2 = arr_float1.astype('int32')
print(arr_int2, arr_int2.dtype)

## ndarray 생성 - arange, zeros, ones
 
# arange()
 - range()와 유사한 기능, 
 - array를 range()로 표현하는 것
 
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

# zeros()
 - 튜플 형태의 shape 값을 입력하면 모든 값을 0으로 채운 ndarray 반환

zero_array = np.zeros((3,2), dtype='int32')
print(zero_array) 
print(zero_array.dtype, zero_array.shape)

# ones()
 - 모든 값을 1로 채운 ndarray로 입력
one_array = np.ones((3,2))
print(one_array.dtype, one_array.shape)

# reshape(), ndarray의 차원과 크기 변경
arr1 = np.arange(10)
print(f'arr1:\n {arr1}')

arr2 = arr1.reshape(2,5)
print(f'arr2:\n {arr2}')

arr3 = arr2.reshape(5,2)
print(f'arr3: \n {arr3}')

# reshape에 -1 적용
 -1 적용시, 원래 ndarray와 호환되는 새로운 shape로 변환
reshape_arr2 = arr1.reshape(-1, 5)
 # 로우 인자 : -1, 컬럼 인자 :5
 # arr1과 호홛될 수 있는 2차원 ndarray로 변환하되, 고정된 5개 컬럼에 맞는 로우를 자동으로 생성해 변환하라는 의미

reshape_arr3 = arr1.reshape(5, -1) 

arr1.reshape(-1,1)
#=> 여러 로우를 갖되, 1개의 컬럼을 가진 ndarray로 구성

arr1 = np.arange(8)
arr3d = arr1.reshape(2,2,2)
print(f'array3d : \n {arr3d.tolist()}')

 # 3차원 ndarray를 2차원 ndarray로 변환
arr3 = arr3d.reshape(-1,1)
print(f'arr3:\n {arr3.tolist()}')
print(f' arr3 shape: {arr3.shape}')

 # 1차원 ndarray를 2차원 ndarray로 변환
arr4 = arr1.reshape(-1,1)
print(f'arr4:\n {arr4.tolist()}')
print(f'arr4 shape : {arr4.shape}')

## indexing : 넘파이의 ndarray의 데이터 셋 선택하기
-> 일부 또는 특정 데이터만 선택할 수 있도록 하는 인덱싱 알아보기
 # 1) 특정 데이터만 추출
 원하는 위치의 인덱스 값을 지정하면 해당 위치의 데이터 반환
 # 2) 슬라이싱
 연속된 ndarray 추출하는 방식(':'기호 사용)
 ex) 1:5 -> 시작 인덱스 1부터 종료 인덱스 4까지 반환
 # 3) 팬시 인덱싱
일정한 인덱싱 집합을 리스트 또는 ndarray 형태로 지정해 해당 위치에 있는 데이터의 ndarray 반환
# 4) 불리언 인덱싱
특정 조건에 해당 여부에따른 True/False값 인덱싱 집합을 기반으로 T위치에 있는 데이터의 ndarray 반환

# 단일 값 추출
# 1-9 까지 1차원 ndarray 생성
arr1 = np.arange(1,10)

arr1d = np.arange(1,10)
arr2d = arr1d.reshape(3,3)
print(arr2d)

print(f'(row=0,col=0) index 가리키는 값 : {arr2d[0,0]}')
# axis=0(로우), axis=1(컬럼)

# 불리언 인덱싱
arr1d = np.arange(start = 1, stop = 10)
arr1d[arr1d >5]
arr1d > 5

# sort()와 argsort() - 행렬의 정렬
 - np.sort(),원 행렬 유지한 채 정렬된 행렬을 반환
 - ndarray.sort(), 행렬 자체를 정렬된 행렬로 반환
 - argsort(), 정렬된 인덱스 반환
 
org_arr = np.array([3,1,9,5])
np.sort(org_arr) 
org_arr.sort() # 행렬 원본 변경

np.sort(org_arr)[::-1] # 내림차순 변경

## 데이터 핸들링 - 판다스 
* Series와 DataFrame의 큰 차이점은 Seires는 컬럼이 하나뿐인 데이터 구조체이고, DataFrame은 컬럼이 여럿인 데이터 구조체이다.

# 데이터 타입, 분포 확인 메서드 - info(), describe()
info() - 총 데이터 건수와 데이터 타입, NULL 갯수 확인
describe() - 분포, 평규느 최댓값등 숫자형 칼럼의 분포도만 조사(object컬럼은 제외)
value_counts() - 데이터 분포 확인하기 위한 함수.
# ex) titanic_df['Pclass'].value_counts()  (Series 타입에서만 사용가능)

# 색인(indexing)
1) .iloc[] # 위치기반 인덱싱
2) .loc[]  # 컬럼기반 인덱싱

# 정렬 , Aggregation 함수, GroupBy 적용
# DataFrame, Series의 정렬 - sort_values()
sort_values() 메서드 파라미터
 - by        (컬럼명, 컬럼명 기준으로 정렬)
 - ascending (False, 내림차순)  
 - inplace   (True, 정렬된 결과 반영)
 
# Aggregation 함수 적용
 - DataFrame에서 바로 aggregation을 호출할 경우 모든 컬럼에 적용한다.
# groupby() 적용
 ex) groupby(by='Pclass'), Pclass컬럼 기준 GroupBy된 객체를 반환한다.

# 결측치 처리하기
NULL # numpy는 NaN

 - NaN은 평균, 총합 등의 함수 연한 시 제외된다
 - NaN확인 API는 isna()
 - 대체하는 API는 fillna()
 
ex) 
df.isna().sum()
df['ab'].fillna('C000') # inplace 메서드를 사용해야 데이터에 반영된다.

# apply lambda 식으로 데이터 가공
 - lambda식은 함수형 프로그래밍을 지원하기 위해 만들어졌다.
ex)

def get_square(a) :
    return a**2
print(f'3의 제곱은 : {get_square(3)}')

lambda_square = lambda x : x**2
print(f'3의 제곱은  : {lambda_square(3)}')

lambda x : x**2
x는 입력인자를 가리킨다. : 인자의 계산식이다(반환값을 의미).

a = [1,2,3]
squares = map(lambda x : x**2, a)
list(squares)

 - lambda 식은 if else를 지원한다.
 주의점) if절의 경우 if식보다 반환 값을 먼저 기술해야한다.
    lambda x : 'Child' if x <= 15 else 'Adult'
    
# https://seaborn.pydata.org/index.html (그래프 예제)