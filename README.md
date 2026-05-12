# 🏦 Credit Score Prediction using Deep Learning

> Kaggle 신용점수 데이터를 활용하여 고객의 금융 상태, 소비 패턴, 월별 변화 흐름을 기반으로 신용점수를 예측한 딥러닝 기반 Tabular AI 프로젝트

---

# 1. 프로젝트 개요

## 📌 프로젝트 목적

* 고객의 금융 상태 및 행동 패턴을 기반으로 신용점수(Credit Score)를 예측
* 단순 수치형 변수뿐 아니라 금융 도메인 기반 파생변수 및 고객별 월별 패턴을 반영
* MLP와 TabNet을 비교하며 Tabular Deep Learning 구조 실험 진행

---

## 📅 프로젝트 기간

* 2026.xx.xx ~ 2026.xx.xx

---

## 🎯 주요 성과

* Baseline MLP Accuracy : 71%
* 최종 Time-aware TabNet Accuracy : **82%+**
* 금융 비율 변수 + 고객별 월별 패턴 요약(diff, mean) 적용을 통해 성능 향상

---

# 2. 기술 스택

## 🐍 Language

```text
Python
```

---

## 📚 Library

### Data & Preprocessing

* Pandas
* Numpy
* Scikit-learn

### Deep Learning

* PyTorch
* PyTorch TabNet

### Visualization

* Matplotlib
* Seaborn

---

# 3. 데이터 정보

## 📂 Dataset

* Source : Kaggle Credit Score Classification Dataset
* 규모 : 100,000 Rows
* 고객 수 : 12,500명
* 기간 : 고객별 8개월 데이터
* Target : Credit_Score

  * Good
  * Standard
  * Poor

---

## 📌 데이터 특징

본 데이터는 동일 고객(Customer_ID)이 8개월 동안 반복 관측된 구조를 가진다.

따라서 단순 단일 시점 데이터로만 처리하지 않고,
고객별 금융 상태 변화 흐름을 반영하기 위한 시계열 기반 파생변수를 추가하였다.

---

# 4. 프로젝트 흐름

```text
EDA
→ 금융 도메인 기반 전처리
→ 파생변수 생성
→ MLP Baseline 구축
→ MLP 구조 개선
→ TabNet 적용
→ 금융 Ratio Feature 추가
→ 고객별 시계열 요약 Feature 추가
→ Feature Selection
→ 최종 모델 도출
```

---

# 5. 데이터 전처리 및 Feature Engineering

## ✅ 1) 로그 변환

금융 데이터 특성상 소득, 잔고, EMI 등의 금액형 변수는 우측 꼬리가 긴 분포를 가지는 경우가 많았다.

따라서 왜도(skewness)를 확인한 뒤, 왜도가 높은 금액형 변수에 한하여 log1p 변환을 적용하였다.

### 적용 변수

* Annual_Income
* Monthly_Inhand_Salary
* Outstanding_Debt
* Total_EMI_per_month
* Amount_invested_monthly
* Monthly_Balance

---

## ✅ 2) Loan Parsing

Type_of_Loan 변수는 unique 값이 6000개 이상 존재하였으나,
실제 loan 종류는 약 10개 수준이었다.

따라서 문자열 parsing을 통해:

* Loan_Count
* has_auto_loan
* has_mortgage_loan
* has_student_loan
* ...

등의 binary feature로 재구성하였다.

---

## ✅ 3) Payment Behaviour 분해

기존 Payment_Behaviour 변수:

```text
High_spent_Small_value_payments
```

형태를:

* Spent_Level
* Payment_Size

두 변수로 분리하였다.

이를 통해 소비 수준과 결제 규모를 독립적으로 학습할 수 있도록 구성하였다.

---

## ✅ 4) 금융 Ratio 파생변수

단순 절대 금액보다 고객의 상대적인 금융 부담 수준이 더 중요하다고 판단하였다.

### 생성 변수

| 변수명                     | 의미             |
| ----------------------- | -------------- |
| Debt_to_Income          | 소득 대비 부채 비율    |
| EMI_to_Income           | 월급 대비 EMI 부담률  |
| Delay_Risk              | 연체 일수 × 연체 횟수  |
| Inv_Ratio               | 월급 대비 투자 비율    |
| Delay_History_Ratio     | 신용 이력 대비 연체 비율 |
| Balance_to_Salary_Ratio | 월급 대비 잔고 비율    |

---

## ✅ 5) 고객별 월별 시계열 요약 변수

본 데이터는 고객별 8개월 데이터 구조를 가지므로,
단순 현재 상태뿐 아니라 금융 상태 변화 흐름을 반영하고자 하였다.

### 생성 변수

* diff1 : 전월 대비 변화량
* mean_by_customer : 고객별 평균 금융 상태

### 적용 변수

* Outstanding_Debt
* Monthly_Balance
* Delay_from_due_date
* Num_of_Delayed_Payment
* Credit_Utilization_Ratio
* Total_EMI_per_month
* Amount_invested_monthly

---

# 6. EDA

## 📊 Income / Balance Distribution

(이미지)

### 해석

* 금융 금액형 변수는 우측 꼬리가 긴 분포를 가짐
* 일부 변수는 극단값이 존재하였으나 금융 데이터 특성상 실제 가능한 값으로 판단
* 제거보다는 로그 변환 및 clipping 전략 적용

---

## 📊 Correlation Analysis

(이미지)

### 해석

* 절대 금액 변수보다 ratio 변수와 연체 관련 변수의 중요도가 높게 나타남
* 일부 mean 기반 시계열 변수는 기존 변수와 높은 상관성을 가짐

---

# 7. 모델링

---

## 🔹 Baseline MLP

### 구조

* Linear
* ReLU
* CrossEntropyLoss

### 결과

* Accuracy : 71%

---

## 🔹 Improved MLP

### 개선 사항

* BatchNorm
* Dropout
* Deep Network
* NAdam Optimizer

### 결과

* Accuracy : xx%

---

## 🔹 TabNet

### 적용 이유

TabNet은 tabular 데이터에 특화된 딥러닝 모델로,
attention 기반 feature selection과 categorical embedding을 지원한다.

특히 범주형 변수가 많은 금융 데이터에 적합하다고 판단하였다.

---

# 8. 실험 로그

| 실험       | 전처리 / Feature Engineering                                     | 모델                      | Accuracy |
| -------- | ------------------------------------------------------------- | ----------------------- | -------- |
| Baseline | 기본 전처리 (Label Encoding + Standard Scaling)                    | MLP                     | 71.0     |
| Exp1     | 로그 변환 + Type_of_Loan Parsing + Loan Count / Loan Flag 생성      | MLP                     | 72.45    |
| Exp2     | Age_Group 파생변수 추가                                             | MLP                     | 72.35    |
| Exp3     | BatchNorm, Dropout, Hidden Layer 확장 등 MLP 구조 개선               |  MLP+            | 72.60    |
| Exp4     | 기존 전처리 기반 TabNet 적용                                           | TabNet                  | 71     |
| Exp5     | Payment_Behaviour 분해 + 금융 파생변수 일부 추가 + Optimizer/Scheduler 조정 |  MLP+            | 71.3    |
| Exp6     | 범주형 변수 Label Encoding 후 TabNet categorical embedding 적용       | TabNet       | 75.9    |
| Exp7     | 금융 비율 기반 파생변수 추가 + TabNet 하이퍼파라미터 튜닝                          |  TabNet+            | 77.41   |
| Exp8     | 금융 ratio 파생변수 최적화 + 기존 최고 성능 TabNet 세팅 적용                     |  TabNet        | 78.5    |
| Exp9     | 고객별 월별 시계열 요약 변수(diff, mean) 추가                               | TabNet       |   82.79    |
| Exp10    | Feature Importance 기반 Feature Selection 및 변수 정제               |  TabNet |          |


---

# 9. 최종 모델 성능

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.82+ |

---

# 10. 주요 인사이트

* 단순 모델 구조 변경보다 금융 도메인 기반 feature engineering이 더 큰 성능 향상을 가져옴
* 고객별 평균 금융 상태(mean)와 최근 변화량(diff)이 신용점수 예측에 유의미하게 작용
* categorical embedding이 범주형 변수 표현에 효과적이었음
* 시계열 요약 feature 추가 이후 가장 큰 성능 향상을 확인

---

# 11. 회고

* 단순 tabular classification이 아닌 “고객 행동 흐름” 관점의 접근이 중요함을 확인
* 딥러닝 모델 구조보다 feature engineering과 데이터 해석이 더 큰 영향을 줄 수 있음을 경험
* TabNet이 범주형 데이터와 금융 데이터에서 강력한 성능을 보이는 것을 확인
