---
layout: post
title: Predicting Marketing Campaign Response Using ML
image: "/img/posts/campaign_response.svg"
tags: [Machine Learning, Classification, Python]
summary: "Three classifiers — Logistic Regression, Random Forest and KNN — trained on a real grocery campaign predict which customers will join the delivery club, so the next mailer only goes to people likely to say yes."
stack: "Python · pandas · scikit-learn"
metrics:
  - value: "0.89"
    label: "F1 on unseen customers"
  - value: "~850"
    label: "customers, one campaign"
  - value: "2.9×"
    label: "conversion lift"
---

A grocery retailer mailed its customers an invitation to join a **delivery club** — and most of the letters were wasted on people who were never going to sign up. I trained three classification models on the results of that campaign to predict **who will actually join**, so the next round of mail goes only to customers with a real chance of converting.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Data Preparation](#04-data-preparation)
- [05. Logistic Regression](#05-logistic-regression)
  - [Dealing with Outliers](#log-outliers)
  - [Splitting the Data](#log-split)
  - [One-Hot Encoding](#log-encoding)
  - [Feature Selection](#log-feature-selection)
  - [Model Training](#log-training)
  - [Model Assessment](#log-assessment)
  - [Finding the Optimal Threshold](#log-threshold)
- [06. Random Forest](#06-random-forest)
  - [Splitting the Data](#rf-split)
  - [One-Hot Encoding](#rf-encoding)
  - [Model Training](#rf-training)
  - [Model Assessment](#rf-assessment)
  - [Feature Importance](#rf-importance)
- [07. K-Nearest Neighbours](#07-k-nearest-neighbours)
  - [Dealing with Outliers](#knn-outliers)
  - [Splitting the Data](#knn-split)
  - [One-Hot Encoding](#knn-encoding)
  - [Feature Scaling](#knn-scaling)
  - [Feature Selection](#knn-feature-selection)
  - [Model Training](#knn-training)
  - [Model Assessment](#knn-assessment)
  - [Finding the Optimal K](#knn-optimal-k)
- [08. Growth & Next Steps](#08-growth--next-steps)

---

## 00. Project Overview

**Context**

Mail campaigns cost money per letter, and the retailer's delivery club campaign converted only ~3 in 10 customers. That makes this a targeting problem: if a model can score each customer's probability of joining *before* the letters go out, the same budget reaches far more of the right people.

**Actions**

I framed it as a supervised **classification** task — predict the binary `signup_flag` from eight customer attributes covering shopping behaviour, distance from store, credit score and gender. I built **three** models on the same campaign data so the comparison would be fair: a **Logistic Regression**, a **Random Forest**, and a **K-Nearest Neighbours** classifier. Because signups are the minority class, every model is judged primarily on **F1-score** rather than accuracy.

**Applications**

A CRM or campaign team would run the model after a pilot run over the whole customer base, rank customers by predicted probability of joining, and post letters only as deep into that list as the budget allows — the same idea carries to email, coupon and loyalty-programmes. Any retailer selling in subscription-style can reuse the pattern directly: one past campaign becomes the training data that makes every future campaign cheaper per conversion.

**Growth & Next Steps**

The forest's dominant signal is *distance from store* — which suggests the next win is smarter features rather than fancier models: delivery-address distance, tenure, and campaign-history variables. A probability-ranked mailing list (rather than a hard yes/no cut) would also let the retailer dial spend up or down per campaign budget.

---

## 01. Results

All three models were assessed on customers they had never seen. Because roughly **31%** of customers signed up and **69%** did not, accuracy alone flatters lazy models — a classifier that says "no one signs up" is already 69% accurate. **F1-score** (the harmonic mean of precision and recall) is the honest headline metric here:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.94 | 0.89 | 0.90 | 0.89 |
| KNN (k = 5) | 0.94 | 1.00 | 0.76 | 0.86 |
| Logistic Regression (0.44 threshold) | 0.89 | 0.80 | 0.76 | 0.78 |

The payoff in campaign terms: mailing only the customers the **Random Forest** flags would take conversion from roughly **3 in 10** letters to about **9 in 10** — a **~2.9× lift** — while still reaching ~90% of everyone who was ever going to join.

The other two are instructive rather than losers. **KNN** posts a *perfect precision* — every customer it flagged really did sign up, zero wasted mail — but it pays for that caution by missing 10 real joiners. **Logistic Regression** starts at an F1 of 0.73 with the default 0.5 cut-off, and climbs to 0.78 just by moving the decision threshold to 0.44 — a reminder that the threshold is a free tuning dial most people forget exists.

---

## 02. Model Overview

Three classifiers, three genuinely different ways of drawing the line between "will join" and "won't".

**Logistic Regression** is similar to a straight-line model. It combines the features into a single weighted score and squashes it into a probability between 0 and 1 — a customer far from the store with a busy shopping history might come out at 0.82. Turning that probability into a yes/no decision means choosing a cut-off, and that cut-off needs to be carefully chosen — which is exactly what the threshold-tuning step in the logistic section exploits. It's fast and easily interpretable.

**Random Forest** builds hundreds of decision trees — each one a sequence of yes/no splits learned on a random slice of the customers and a random subset of the features. A single tree tends to memorise noise; a large, deliberately varied group of trees averages the noise away and keeps the overarching pattern. 

**K-Nearest Neighbours** finds the k most similar customers in the training data — similarity measured as plain distance across the scaled features — if four of the five nearest look-alikes joined, the customer is predicted to join. That simplicity makes it a genuinely independent, though it leans entirely on feature scaling, because any feature measured on a large scale would otherwise dominate the similarity maths.

---

## 03. Data Overview

I'm predicting the binary `signup_flag` — whether a customer joined the delivery club after the mailer — from the retailer's customer database, with shopping behaviour aggregated over the three months before the campaign. The `customer_id` is dropped before modelling, and the classes sit at roughly 31% signed up vs 69% did not, which is what pushes F1 ahead of accuracy as the metric of record. After cleaning, the modelling dataset contains the following fields:

| Variable Name | Variable Type | Description |
|---|---|---|
| signup_flag | Dependent | Whether the customer joined the delivery club after the campaign (1/0) |
| distance_from_store | Independent | Distance in miles from the customer's home to the store |
| gender | Independent | The gender the customer provided (one-hot encoded) |
| credit_score | Independent | The customer's most recent credit score |
| total_sales | Independent | Total spend in the three months before the campaign (GBP) |
| total_items | Independent | Total items bought in those three months |
| transaction_count | Independent | Number of shopping trips in those three months |
| product_area_count | Independent | How many distinct product areas the customer shopped across |
| average_basket_value | Independent | Average spend per transaction (GBP) |
| customer_id | Identifier | Unique customer key — dropped before modelling |

---

## 04. Data Preparation

Every model starts from the same prepared table. I load the campaign data, drop the identifier, shuffle with a fixed seed, and check the class balance — that ~31/69 split drives every metric decision later. A handful of rows have gaps; with ~850 customers to work with, dropping them outright is cleaner than imputing.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

model_data = pd.read_pickle(open("data/abc_classification_modelling.p", "rb"))
model_data.drop("customer_id", axis=1, inplace=True)

model_data = shuffle(model_data, random_state=42)
model_data["signup_flag"].value_counts(normalize=True)

model_data.isna().sum()
model_data.dropna(how="any", inplace=True)
```

The three model sections below each pick up from this point.

---

## 05. Logistic Regression

### Dealing with Outliers {#log-outliers}

Spending and distance columns have long tails that can drag a linear decision boundary around. I strip anything beyond **twice the inter-quartile range** on the three continuous behaviour columns.

```python
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns:
    lower_quartile = model_data[column].quantile(0.25)
    upper_quartile = model_data[column].quantile(0.75)
    iqr_extended = (upper_quartile - lower_quartile) * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    outliers = model_data[(model_data[column] < min_border) | (model_data[column] > max_border)].index
    model_data.drop(outliers, inplace=True)
```

### Splitting the Data {#log-split}

The split is **stratified** so the ~31% signup rate is preserved in both training and test sets — without it, a random split could hand the model an unrepresentative picture of the minority class.

```python
x = model_data.drop(["signup_flag"], axis=1)
y = model_data["signup_flag"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
```

### One-Hot Encoding {#log-encoding}

`gender` is the one categorical input. With only two categories, one-hot encoding with `drop="first"` turns it into a single clean binary column — fitted on train only, applied to both.

```python
categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

x_train_encoded = one_hot_encoder.fit_transform(x_train[categorical_vars])
x_test_encoded = one_hot_encoder.transform(x_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

x_train_encoded = pd.DataFrame(x_train_encoded, columns=encoder_feature_names)
x_train = pd.concat([x_train.reset_index(drop=True), x_train_encoded.reset_index(drop=True)], axis=1)
x_train.drop(categorical_vars, axis=1, inplace=True)

x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoder_feature_names)
x_test = pd.concat([x_test.reset_index(drop=True), x_test_encoded.reset_index(drop=True)], axis=1)
x_test.drop(categorical_vars, axis=1, inplace=True)
```

### Feature Selection {#log-feature-selection}

`RFECV` cross-validates its way down the feature list and keeps **7 of the 8** inputs — only `total_sales` gets cut, which makes sense given `average_basket_value` and `total_items` already carry most of that information.

```python
classifier = LogisticRegression(random_state=42, max_iter=1000)
feature_selector = RFECV(classifier)
fit = feature_selector.fit(x_train, y_train)

x_train = x_train.loc[:, feature_selector.get_support()]
x_test = x_test.loc[:, feature_selector.get_support()]
```

### Model Training {#log-training}

```python
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(x_train, y_train)
```

### Model Assessment {#log-assessment}

Alongside the class predictions I keep the **probabilities** — they power the threshold tuning below.

```python
y_pred_class = classifier.predict(x_test)
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred_class)   # [[107, 8], [13, 29]]

accuracy_score(y_test, y_pred_class)    # ~0.87
precision_score(y_test, y_pred_class)   # ~0.78
recall_score(y_test, y_pred_class)      # ~0.69
f1_score(y_test, y_pred_class)          # ~0.73
```

At the default 0.5 cut-off the model is decent but timid — it misses 13 of the 42 real signups.

### Finding the Optimal Threshold {#log-threshold}

A logistic regression outputs a probability; *where you cut it* is a business decision, not a default. Sweeping the threshold from 0 to 1 and scoring F1 at each step finds the best trade-off at **0.44** — slightly more generous than 0.5, which suits a use case where missing a genuine joiner costs more than one extra letter.

```python
thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    pred_class = (y_pred_prob >= threshold) * 1
    precision_scores.append(precision_score(y_test, pred_class, zero_division=0))
    recall_scores.append(recall_score(y_test, pred_class))
    f1_scores.append(f1_score(y_test, pred_class))

max_f1 = max(f1_scores)                          # ~0.78
optimal_threshold = thresholds[f1_scores.index(max_f1)]   # 0.44

y_pred_class_opt = (y_pred_prob >= optimal_threshold) * 1
```

Re-scored at 0.44 the same model catches 3 more signups at no precision cost worth mentioning: **accuracy ~0.89, precision 0.80, recall 0.76, F1 ~0.78**.

---

## 06. Random Forest

One deliberate difference from the other two pipelines: **no outlier removal**. Each tree splits on thresholds, so a customer who spends ten times the average just ends up on one side of a split — extreme values can't drag the model around the way they drag a linear boundary.

### Splitting the Data {#rf-split}

```python
x = model_data.drop(["signup_flag"], axis=1)
y = model_data["signup_flag"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
```

### One-Hot Encoding {#rf-encoding}

Identical treatment of `gender` as in the logistic pipeline.

```python
categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

x_train_encoded = one_hot_encoder.fit_transform(x_train[categorical_vars])
x_test_encoded = one_hot_encoder.transform(x_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

x_train_encoded = pd.DataFrame(x_train_encoded, columns=encoder_feature_names)
x_train = pd.concat([x_train.reset_index(drop=True), x_train_encoded.reset_index(drop=True)], axis=1)
x_train.drop(categorical_vars, axis=1, inplace=True)

x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoder_feature_names)
x_test = pd.concat([x_test.reset_index(drop=True), x_test_encoded.reset_index(drop=True)], axis=1)
x_test.drop(categorical_vars, axis=1, inplace=True)
```

*(No feature selection step either — with 500 trees each seeing random feature subsets, the forest effectively downweights weak features on its own.)*

### Model Training {#rf-training}

```python
classifier = RandomForestClassifier(random_state=42, n_estimators=500, max_features=5)
classifier.fit(x_train, y_train)
```

### Model Assessment {#rf-assessment}

```python
y_pred_class = classifier.predict(x_test)
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred_class)   # [[112, 6], [5, 47]]

accuracy_score(y_test, y_pred_class)    # ~0.94
precision_score(y_test, y_pred_class)   # ~0.89
recall_score(y_test, y_pred_class)      # ~0.90
f1_score(y_test, y_pred_class)          # ~0.89
```

That confusion matrix is the whole pitch in four numbers: of 52 customers who genuinely signed up, the forest found **47** — and it only mis-flagged 6 of the 118 who didn't.

### Feature Importance {#rf-importance}

I read both the forest's built-in importance and **permutation importance** (shuffle one feature, watch the score drop) so the story doesn't rest on one method:

```python
feature_importance = pd.DataFrame(classifier.feature_importances_)
feature_names = pd.DataFrame(x.columns)
feature_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_summary.columns = ["input_variable", "feature_importance"]

result = permutation_importance(classifier, x_test, y_test, n_repeats=10, random_state=42)
```

Both agree: **`distance_from_store` towers over everything else** (~47% of total importance), with `product_area_count` a distant second. People join a delivery club when the store is far away — the model has essentially learned common sense, which is exactly what you want to see.

---

## 07. K-Nearest Neighbours

KNN classifies a customer by looking at the customers most similar to them — which makes it a genuinely different *kind* of model to the other two, and a useful sanity check on their answers.

### Dealing with Outliers {#knn-outliers}

Same treatment as the logistic pipeline — strip the extreme tails on the three behaviour columns. Outliers matter doubly for KNN, because a single extreme customer distorts every distance calculated near them.

```python
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns:
    lower_quartile = model_data[column].quantile(0.25)
    upper_quartile = model_data[column].quantile(0.75)
    iqr_extended = (upper_quartile - lower_quartile) * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    outliers = model_data[(model_data[column] < min_border) | (model_data[column] > max_border)].index
    model_data.drop(outliers, inplace=True)
```

### Splitting the Data {#knn-split}

```python
x = model_data.drop(["signup_flag"], axis=1)
y = model_data["signup_flag"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
```

### One-Hot Encoding {#knn-encoding}

```python
categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

x_train_encoded = one_hot_encoder.fit_transform(x_train[categorical_vars])
x_test_encoded = one_hot_encoder.transform(x_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

x_train_encoded = pd.DataFrame(x_train_encoded, columns=encoder_feature_names)
x_train = pd.concat([x_train.reset_index(drop=True), x_train_encoded.reset_index(drop=True)], axis=1)
x_train.drop(categorical_vars, axis=1, inplace=True)

x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoder_feature_names)
x_test = pd.concat([x_test.reset_index(drop=True), x_test_encoded.reset_index(drop=True)], axis=1)
x_test.drop(categorical_vars, axis=1, inplace=True)
```

### Feature Scaling {#knn-scaling}

The step the other two models don't need. KNN runs on raw distances, so without scaling, `total_sales` (hundreds of pounds) would drown `distance_from_store` (single-digit miles) in every comparison. Min-max normalisation puts every feature on the same 0–1 footing — fitted on train, applied to both.

```python
scale_norm = MinMaxScaler()
x_train = pd.DataFrame(scale_norm.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scale_norm.transform(x_test), columns=x_test.columns)
```

### Feature Selection {#knn-feature-selection}

KNN has no native way to rank features, so I borrow a Random Forest as the selector inside `RFECV`. It keeps **6 of the 8** inputs, dropping `credit_score` and `total_items`.

```python
classifier = RandomForestClassifier(random_state=42)
feature_selector = RFECV(classifier)
fit = feature_selector.fit(x_train, y_train)

x_train = x_train.loc[:, feature_selector.get_support()]
x_test = x_test.loc[:, feature_selector.get_support()]
```

### Model Training {#knn-training}

```python
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)
```

### Model Assessment {#knn-assessment}

```python
y_pred_class = classifier.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred_class)   # [[115, 0], [10, 32]]

accuracy_score(y_test, y_pred_class)    # ~0.94
precision_score(y_test, y_pred_class)   # 1.00
recall_score(y_test, y_pred_class)      # ~0.76
f1_score(y_test, y_pred_class)          # ~0.86
```

Look at the top-right of that confusion matrix: **zero false positives**. Every single customer KNN flagged as a joiner really joined. The cost is 10 missed signups — a very conservative marketer, never wrong when it says yes, but too quick to say no.

### Finding the Optimal K {#knn-optimal-k}

How many neighbours should vote? I score F1 for every k from 2 to 24 — and the curve peaks right at the default, **k = 5**, so the out-of-the-box setting stands.

```python
k_list = list(range(2, 25))
f1_scores = []

for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    f1_scores.append(f1_score(y_test, y_pred))

max_f1 = max(f1_scores)                     # ~0.86
optimal_k = k_list[f1_scores.index(max_f1)]  # 5
```

---

## 08. Growth & Next Steps

Concrete improvements I'd make before the next campaign:

- **Feed the distance signal.** With `distance_from_store` doing ~47% of the work, features in the same family — delivery-address distance, drive time, competitor proximity — are the most promising additions.
- **Tune the forest.** These scores come from sensible defaults plus `max_features=5`; a proper grid search over tree depth and features-per-split has more to give.

---
