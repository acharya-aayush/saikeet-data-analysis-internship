
# 01_data_preparation_notes.md

## Step-by-step Guide (for newbies, by a newbie)

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style='whitegrid')
```
- Why: pandas for data, numpy for numbers, matplotlib/seaborn for plots.
- Source: pandas docs, matplotlib docs, seaborn docs.

### 2. Loading & Inspecting Data
```python
df = pd.read_csv('../data/raw/Telco_Customer_Churn_Dataset  (3).csv')
df.head()
df.info()
df.isnull().sum()
```
- Why: See what the data looks like, check for missing values.
- Output: Shows first 5 rows, info about columns, missing values per column.
- Source: pandas.read_csv, pandas.DataFrame.head/info/isnull

### 3. Cleaning Data
```python
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df = df.drop_duplicates()
df.dtypes
if 'TotalCharges' in df.columns:
	df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
```
- Why: Remove dups, fix types, drop missing rows (could also fillna, but dropped for now).
- Output: Number of dups, dtypes, cleaned df.
- Source: pandas.DataFrame.duplicated/drop_duplicates/dtypes/to_numeric/dropna

### 4. EDA (Exploratory Data Analysis)
```python
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()
plt.figure(figsize=(8,4))
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title('Distribution of Customer Tenure (Months)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(8,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn Status')
plt.show()
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
plt.show()
```
- Why: Visualize churn, tenure, charges, gender.
- Output: Plots show churn balance, tenure spread, charges by churn, gender balance.
- Source: seaborn.countplot/histplot/boxplot, matplotlib.pyplot

### 5. Stats Summary
```python
df.describe()
df.describe(include='object')
```
- Why: Quick stats for numbers and categories.
- Output: Means, std, min/max, unique counts, etc.
- Source: pandas.DataFrame.describe

### 6. Encoding Categoricals & Splitting Data
```python
from sklearn.model_selection import train_test_split
cat_cols = df.select_dtypes(include='object').columns.drop('customerID') if 'customerID' in df.columns else df.select_dtypes(include='object').columns
for col in cat_cols:
	df[col] = df[col].astype('category').cat.codes
y = df['Churn']
X = df.drop(['Churn','customerID'], axis=1, errors='ignore')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Why: ML needs numbers, not strings. Split for training/testing.
- Output: Encoded df, X_train/X_test/y_train/y_test ready for ML.
- Source: pandas.DataFrame.astype/cat.codes, scikit-learn train_test_split

---

## What the Output Showed
- Data loaded fine, no major errors.
- Some dups found (if any), dropped.
- 'TotalCharges' sometimes needs conversion to numeric (was string for some rows).
- Dropped rows with missing values (could try filling next time).
- Churn is imbalanced (more non-churn than churn).
- Tenure is right-skewed (most customers are newer).
- Monthly charges: churners pay a bit more on avg.
- Gender is balanced.
- After encoding, all features are numeric.
- Data split 80/20 for ML.

---

## Sources Used
- pandas docs: https://pandas.pydata.org/docs/
- seaborn docs: https://seaborn.pydata.org/
- matplotlib docs: https://matplotlib.org/stable/contents.html
- scikit-learn docs: https://scikit-learn.org/stable/
- Some StackOverflow for .cat.codes and train_test_split usage

---

## Why These Choices
- Used .cat.codes for quick label encoding (not best for all ML, but simple for now)
- Dropped missing rows for simplicity (could impute/fill in real projects)
- Used train_test_split with random_state=42 for reproducibility
- Plots chosen to quickly see class balance, distributions, and relationships

---

## What I'd Do Next
- More EDA: churn by contract/payment/partner/dependents
- Try segmenting customers
- Start basic ML models (logreg, tree)

......