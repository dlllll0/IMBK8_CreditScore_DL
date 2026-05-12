# 🏦 Credit Score Prediction using Deep Learning

> Kaggle 신용점수 데이터를 활용하여 고객의 금융 상태, 소비 패턴, 월별 변화 흐름을 기반으로 신용점수를 예측한 딥러닝 기반 Tabular AI 프로젝트

---

# 1. 프로젝트 개요

## 📌 프로젝트 목적

- 고객의 금융 상태 및 행동 패턴을 기반으로 신용점수(Credit Score)를 예측
- 단순 수치형 변수뿐 아니라 금융 도메인 기반 파생변수 및 고객별 월별 패턴을 반영
- MLP와 TabNet을 비교하며 Tabular Deep Learning 구조 실험 진행

---

## 📅 프로젝트 기간

- 2026.05.12

---

## 🎯 주요 성과

- Baseline MLP Accuracy : 71%
- 최종 Deep MLP Accuracy : **83.36%**
- 금융 Ratio 변수 + 고객별 월별 패턴 요약(diff, mean) 적용을 통해 성능 향상

---

# 2. 기술 스택

## 🐍 Language

```text
Python
```

---

## 📚 Library

### Data & Preprocessing

- Pandas
- NumPy
- Scikit-learn

### Deep Learning

- PyTorch
- PyTorch TabNet

### Visualization

- Matplotlib
- Seaborn

---

# 3. 데이터 정보

## 📂 Dataset

- Source : Kaggle Credit Score Classification Dataset
- 규모 : 100,000 Rows
- 고객 수 : 12,500명
- 기간 : 고객별 8개월 데이터
- Target : Credit_Score

| Target | 의미 |
|---|---|
| Good | 높은 신용등급 |
| Standard | 일반 신용등급 |
| Poor | 낮은 신용등급 |

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
→ Deep MLP 튜닝
→ 최종 모델 선정
```

---

# 5. EDA

## 📊 Credit Score 분포

(이미지 첨부)

### 해석

- Standard 클래스 비율이 가장 높게 나타남
- 클래스 비율 차이가 존재하여 train / validation split 시 stratify 적용

---

## 📊 금액형 변수 분포

(이미지 첨부)

### 해석

Annual_Income, Outstanding_Debt, Monthly_Balance 등의 금액형 변수는 우측 꼬리가 긴 분포를 보였다.

금융 데이터 특성상 큰 값 자체가 의미를 가질 수 있다고 판단하여 이상치 제거보다는 로그 변환을 적용하였다.

### 적용 변수

- Annual_Income
- Monthly_Inhand_Salary
- Outstanding_Debt
- Total_EMI_per_month
- Amount_invested_monthly
- Monthly_Balance

---

## 📊 Outstanding_Debt와 Credit Score 관계

(이미지 첨부)

### 해석

Outstanding_Debt가 높은 고객일수록 Poor 등급 비율이 높게 나타났다.

단순 부채 금액보다 소득 대비 부채 부담이 더 중요하다고 판단하여 다음 파생변수를 생성하였다.

```python
Debt_to_Income = Outstanding_Debt / Annual_Income
```

---

## 📊 Delay 관련 변수 분석

(이미지 첨부)

### 해석

- Delay_from_due_date
- Num_of_Delayed_Payment

두 변수 모두 신용등급과 관련성이 높게 나타났다.

특히 연체일수와 연체횟수가 동시에 증가할 경우 위험도가 더 커질 수 있다고 판단하여 다음 변수를 추가하였다.

```python
Delay_Risk = Delay_from_due_date * Num_of_Delayed_Payment
```

---

## 📊 Correlation Analysis

(이미지 첨부)

### 해석

상관계수 분석은 변수 제거 목적보다는 변수 간 관계를 파악하고 파생변수 생성 아이디어를 얻기 위한 용도로 활용하였다.

본 프로젝트는 딥러닝 기반 모델을 사용하였기 때문에, 선형 모델처럼 상관계수만을 기준으로 변수 제거를 진행하지는 않았다.

---

# 6. 데이터 전처리 및 Feature Engineering

## ✅ 1) 식별자 제거

다음 변수들은 모델이 고객 자체를 외우는 방향으로 학습할 가능성이 있다고 판단하여 제거하였다.

- ID
- Customer_ID
- Name
- SSN

단, `Customer_ID`는 고객별 월별 패턴 생성을 위한 group 기준으로만 사용하였다.

---

## ✅ 2) 로그 변환

금융 데이터 특성상 소득, 잔고, EMI 등의 금액형 변수는 우측 꼬리가 긴 분포를 가지는 경우가 많았다.

따라서 왜도(skewness)를 확인한 뒤, 왜도가 높은 금액형 변수에 한하여 `log1p` 변환을 적용하였다.

### 적용 변수

- Annual_Income
- Monthly_Inhand_Salary
- Outstanding_Debt
- Total_EMI_per_month
- Amount_invested_monthly
- Monthly_Balance

---

## ✅ 3) Loan Parsing

Type_of_Loan 변수는 unique 값이 6000개 이상 존재하였으나, 실제 loan 종류는 약 10개 수준이었다.

예시:

```text
Auto Loan, Mortgage Loan, Student Loan
```

따라서 문자열 parsing을 통해 다음과 같은 binary feature로 재구성하였다.

### 생성 변수

- Loan_Count
- has_auto_loan
- has_mortgage_loan
- has_student_loan
- has_personal_loan
- has_payday_loan
- has_credit_builder_loan

조합형 문자열을 그대로 encoding 하는 것보다 안정적인 성능을 보였다.

---

## ✅ 4) Payment Behaviour 분해

기존 Payment_Behaviour 변수:

```text
High_spent_Small_value_payments
```

형태를 다음 두 변수로 분리하였다.

- Spent_Level
- Payment_Size

소비 수준과 결제 규모를 독립적으로 학습할 수 있도록 구성하였다.

---

## ✅ 5) Age_Group 생성

Age 변수는 그대로 유지하면서 추가적으로 Age_Group 파생변수를 생성하였다.

연령대별 금융 패턴 차이가 존재할 수 있다고 판단하였다.

---

## ✅ 6) 금융 Ratio 파생변수

프로젝트에서 가장 성능 향상에 큰 영향을 준 부분이다.

단순 절대 금액보다 고객의 상대적인 금융 부담 수준을 표현하는 데 집중하였다.

### 생성 변수

| 변수명 | 의미 |
|---|---|
| Debt_to_Income | 소득 대비 부채 비율 |
| EMI_to_Income | 월급 대비 EMI 부담률 |
| Delay_Risk | 연체 위험도 |
| Inv_Ratio | 투자 비율 |
| Delay_History_Ratio | 신용 이력 대비 연체 비율 |
| Balance_to_Salary_Ratio | 월급 대비 잔고 비율 |

### 생성 이유

#### Debt_to_Income

같은 부채 금액이라도 소득 수준에 따라 부담 정도가 다를 수 있기 때문에 추가하였다.

#### EMI_to_Income

월급 대비 EMI 비율을 통해 고객의 현금흐름 부담을 반영하고자 하였다.

#### Delay_Risk

연체일수와 연체횟수가 동시에 높을 경우 위험도가 더 커질 수 있다고 판단하였다.

#### Delay_History_Ratio

신용이력이 짧은 고객의 연체는 더 위험한 신호일 수 있다고 판단하였다.

---

## ✅ 7) 고객별 월별 시계열 요약 변수

본 데이터는 고객별 8개월 데이터 구조를 가지므로, 단순 현재 상태뿐 아니라 금융 상태 변화 흐름을 반영하고자 하였다.

### 생성 변수

- diff1 : 전월 대비 변화량
- mean_by_customer : 고객별 평균 금융 상태

### 적용 변수

- Outstanding_Debt
- Monthly_Balance
- Delay_from_due_date
- Num_of_Delayed_Payment
- Credit_Utilization_Ratio
- Total_EMI_per_month
- Amount_invested_monthly

### 생성 이유

#### diff1

전월 대비 부채 증가, 잔고 감소, 연체 증가 등의 변화 흐름을 반영하기 위해 추가하였다.

#### mean_by_customer

고객 평균 금융 상태 자체가 장기적인 신용 패턴을 반영할 수 있다고 판단하였다.

---

## ✅ 8) 추가 파생변수

최종 성능 개선을 위해 다음 변수를 추가하였다.

| 변수명 | 의미 |
|---|---|
| Estimated_Spending | 월별 소비 추정 |
| Spending_to_Income | 소비 비율 |
| Inquiry_to_History | 신용조회 대비 신용이력 |
| Interest_Debt_Risk | 이자율 × 부채 위험 |

---

# 7. 모델링

## 🔹 Baseline MLP

### 구조

- Linear
- ReLU
- CrossEntropyLoss

### 결과

- Accuracy : 71%

---

## 🔹 Improved MLP

### 개선 사항

- BatchNorm
- Dropout
- Hidden Layer 확장
- NAdam Optimizer
- ReduceLROnPlateau
- EarlyStopping

### 결과

- Accuracy : 83.36%

---

## 🔹 TabNet

### 적용 이유

TabNet은 tabular 데이터에 특화된 딥러닝 모델로,  
attention 기반 feature selection과 categorical embedding을 지원한다.

범주형 변수가 많은 금융 데이터에 적합하다고 판단하여 적용하였다.

---

# 8. 실험 로그

| 실험 | 전처리 / Feature Engineering | 모델 | Accuracy |
|---|---|---|---|
| Baseline | 기본 전처리 (Label Encoding + Standard Scaling) | MLP | 71.0 |
| Exp1 | 로그 변환 + Type_of_Loan Parsing + Loan Count / Loan Flag 생성 | MLP | 72.45 |
| Exp2 | Age_Group 파생변수 추가 | MLP | 72.35 |
| Exp3 | BatchNorm, Dropout, Hidden Layer 확장 등 MLP 구조 개선 | MLP+ | 72.60 |
| Exp4 | 기존 전처리 기반 TabNet 적용 | TabNet | 71.0 |
| Exp5 | Payment_Behaviour 분해 + 금융 파생변수 일부 추가 | MLP+ | 71.3 |
| Exp6 | 범주형 변수 Label Encoding 후 TabNet categorical embedding 적용 | TabNet+ | 75.9 |
| Exp7 | 금융 비율 기반 파생변수 추가 + TabNet 튜닝 | TabNet++ | 77.41 |
| Exp8 | 금융 ratio 변수 최적화 | TabNet | 78.5 |
| Exp9 | 고객별 월별 시계열 요약 변수(diff, mean) 추가 | TabNet++ | 82.79 |
| Exp10 | Feature Importance 기반 Feature Selection | TabNet FS | 81.46 |
| Exp11 | Deep MLP 튜닝 + EarlyStopping | MLP++ | 83.14 |
| Exp12 | 추가 금융 변수 적용 + Deep MLP | MLP Final | **83.36** |

---

# 9. Feature Importance

(이미지 첨부)

### 해석

- Month
- Credit_Mix
- Interest_Rate
- 연체 관련 변수

등이 높은 중요도를 보였다.

특히 고객별 mean / diff 기반 변수들이 성능 향상에 크게 기여하였다.

---

# 10. 최종 모델 성능

| Metric | Score |
|---|---|
| Accuracy | **0.8336** |

---

# 11. 주요 인사이트

- 단순 모델 구조 변경보다 금융 도메인 기반 Feature Engineering이 더 큰 성능 향상을 가져왔다.
- 고객별 평균 금융 상태(mean)와 최근 변화량(diff)이 신용점수 예측에 유의미하게 작용하였다.
- categorical embedding이 범주형 변수 표현에 효과적이었다.
- 시계열 요약 feature 추가 이후 가장 큰 성능 향상을 확인할 수 있었다.

---

# 12. 회고

- 단순 tabular classification이 아닌 고객 행동 흐름 관점의 접근이 중요함을 확인할 수 있었다.
- 딥러닝 모델 구조보다 feature engineering과 데이터 해석이 더 큰 영향을 줄 수 있음을 경험하였다.
- 충분한 Feature Engineering 이후에는 Deep MLP 역시 tabular classification에서 강력한 성능을 보일 수 있다는 점을 확인하였다.
