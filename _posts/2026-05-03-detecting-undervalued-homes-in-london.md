---
layout: post
title: Detecting Undervalued Homes in London Using ML
image: "/img/posts/undervalued_homes.svg"
tags: [Machine Learning, Regression, Python]
summary: "Two regression models trained on ~6,300 real London sales predict a fair price for every home — then a residual-based detector flags the ones that sold 50%+ below what they should have."
stack: "Python · pandas · scikit-learn"
metrics:
  - value: "0.76"
    label: "R² on unseen sales"
  - value: "~6,300"
    label: "London sales, 2024"
  - value: "~40"
    label: "homes flagged as undervalued"
---

I built a model that predicts a London property's **fair sale price** from its size, location and type — then used the user's **threshold (set at 50%)** to flag the homes that sold for far **less** than they should have. In other words, turning a regression model into a bargain-hunting shortlist.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Linear Regression](#04-linear-regression)
  - [Data Import](#lin-data-import)
  - [Dealing with Missing Values](#lin-missing)
  - [Dealing with Outliers](#lin-outliers)
  - [Splitting the Data](#lin-split)
  - [Target Encoding](#lin-encoding)
  - [Feature Selection](#lin-feature-selection)
  - [Model Training](#lin-training)
  - [Model Assessment](#lin-assessment)
- [05. Random Forest](#05-random-forest)
  - [Data Import](#rf-data-import)
  - [Dealing with Missing Values](#rf-missing)
  - [Dealing with Outliers](#rf-outliers)
  - [Splitting the Data](#rf-split)
  - [Target Encoding](#rf-encoding)
  - [Model Training](#rf-training)
  - [Model Assessment](#rf-assessment)
- [06. Detecting Undervalued Homes](#06-detecting-undervalued-homes)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

"Was this house a good deal?". A listing price tells you what the seller *wants*; it doesn't tell you what the home is *worth*. I wanted to see whether a model trained on thousands of real London sales could estimate a fair price accurately enough to spot the outliers — homes that changed hands well below their predicted value.

**Actions**

I trained two regression models on ~6,300 London sales from 2024 to predict total sale price from three features — floor area, property type, and location (postcode area). I compared a **Linear Regression** and a **Random Forest**, then built a residual-based detector on top to flag and name the most undervalued homes.

**Applications**

This is essentially a buy-side screening tool for anyone who works through property lists. A property investor or buy-to-let landlord could run each new batch of listings through the model and only spend viewing time on homes priced well below their predicted value, while an estate agency could use the same logic in reverse to spot under-priced instructions before they go live. 

**Growth & Next Steps**

The flag is a *shortlist to investigate, not a definitive bargain* — the model isn't trained on condition, or any further details about the property. Natural next steps are an upper-bound guard on the threshold, a fixed random seed for a reproducible shortlist, and cross-checking flags against the dataset's own sale estimates.

---

## 01. Results

I framed this as a supervised regression problem: predict total `history_price` from floor area, property type, and (target-encoded) location. I deliberately built **two** models on the *same* cleaned data — a fully interpretable **Linear Regression** and a non-linear **Random Forest** — so the comparison would be fair. Both followed the same spine: shuffle → handle missing values → remove outliers → train/test split → target-encode → train → assess.

Here is how they scored:

| Model | Adjusted R-Squared (Test Set) | R-Squared (K-Fold CV, k = 4) |
|---|---|---|
| Random Forest | 0.76 | 0.73 |
| Linear Regression | 0.75 | 0.74 |

The 500-tree forest **only slightly increased** the model's predictability compared to the simple linear model. They roughly land within a whisker of each other (~0.75), and across different random splits the lead swaps back and forth.

The two models also **agree on the output that matters**: run the undervaluation detector on each and they independently flag almost the same set — 41 homes (linear) and 42 homes (forest) sold 50%+ below prediction, out of 1,253 test properties. When two different models point at the same ~40 houses, the shortlist is worth trusting a little more.

---

## 02. Model Overview

This project solves the same task two different ways, which makes it a natural place to see what each model actually is.

**Linear Regression** is the simplest model and assumes each feature pushes the price up or down by a fixed amount — every extra square metre adds roughly the same number of pounds — and fits the straight line through the training data that produces the smallest overall error. The result is fully transparent: you can read the coefficients directly and say exactly why the model priced a home the way it did. The trade-off is rigidity — if the true relationship curves, or features interact, a straight line can't bend to follow it.

**Random Forest** attacks the problem with hundreds of decision trees instead. Each tree learns a sequence of yes/no splits on a random sample of the data and a random subset of the features, and the forest's prediction is the average of all trees' answers. The randomness makes each tree work in a slightly different way, and averaging washes individual errors out. Forests capture curved relationships and feature interactions natively and are hard to throw off with odd values — however, you are losing the line-by-line interpretability of the linear model.

Here the forest's extra flexibility bought almost nothing (~0.76 vs ~0.75 R²) — a useful finding in itself, because it says these three features relate to price in a mostly linear way.

---

## 03. Data Overview

I'm predicting the continuous `history_price` — the total price a London home actually sold for in 2024 — from the Kaggle London house-price dataset. The raw file was filtered to 2024 sales only and de-duplicated, `Price_per_SqM` was derived purely as an outlier-removal tool, and `fullAddress` was set aside before modelling so it could label the final shortlist without ever becoming a feature. After this pre-processing, the modelling dataset contains the following fields:

| Variable Name | Variable Type | Description |
|---|---|---|
| history_price | Dependent | The total price the home sold for in 2024 (GBP) |
| outcode | Independent | The postcode area (e.g. SW11) — the location signal, ~141 categories, target-encoded |
| floorAreaSqM | Independent | The internal floor area of the home in square metres |
| propertyType | Independent | The type of home — flat, terraced, semi-detached, detached, etc. |
| Price_per_SqM | Derived | Sale price ÷ floor area — used only to strip outliers, dropped before modelling |
| fullAddress | Identifier | The full property address — held out of the features, re-attached at the end to name the shortlist |

---

## 04. Linear Regression

### Data Import {#lin-data-import}

I load the Kaggle London file, keep only 2024 sales, de-duplicate, and derive `Price_per_SqM` (used later only for outlier removal). Crucially, `fullAddress` is set **aside before modelling** so it can never leak into the features — but can still label the results at the end.

```python
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

model_data = pd.read_csv("kaggle_london_house_price_data.csv")
model_data["history_date"] = pd.to_datetime(model_data["history_date"], errors="coerce")
model_data = model_data[model_data["history_date"].dt.year == 2024]
model_data = model_data.drop_duplicates(subset=["fullAddress", "history_price"])
model_data["Price_per_SqM"] = model_data["history_price"] / model_data["floorAreaSqM"]

address_lookup = model_data["fullAddress"].copy()
model_data = model_data[["outcode", "floorAreaSqM", "propertyType", "history_price", "Price_per_SqM"]]
model_data = shuffle(model_data)
```

### Dealing with Missing Values {#lin-missing}

A row missing a floor area or price is useless for this model, so I drop any incomplete rows outright.

```python
model_data.dropna(how="any", inplace=True)
```

### Dealing with Outliers {#lin-outliers}

London price-per-square-metre has a long tail of data-entry errors and non-standard sales. I remove anything beyond **twice the inter-quartile range** on `Price_per_SqM` — wide enough to keep genuine variety, tight enough to strip the noise.

```python
for column in ["Price_per_SqM"]:
    lower_quartile = model_data[column].quantile(0.25)
    upper_quartile = model_data[column].quantile(0.75)
    iqr_extended = (upper_quartile - lower_quartile) * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    outliers = model_data[(model_data[column] < min_border) | (model_data[column] > max_border)].index
    model_data.drop(outliers, inplace=True)
```

### Splitting the Data {#lin-split}

`Price_per_SqM` was only ever an outlier tool, so it's dropped here — the model predicts total `history_price` from location, floor area and property type.

```python
x = model_data.drop(["Price_per_SqM", "history_price"], axis=1)
y = model_data["history_price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### Target Encoding {#lin-encoding}

Postcode area (`outcode`) has ~141 categories — one-hot encoding would explode that into 140 sparse columns and choke on unseen areas. Instead I **target-encode**: each area becomes a single number, the typical price for that area. The mapping is learned on **train only**, then applied to both sets.

```python
categorical_vars = ["propertyType", "outcode"]
target_encoder = TargetEncoder(target_type="continuous")

x_train[categorical_vars] = target_encoder.fit_transform(x_train[categorical_vars], y_train)
x_test[categorical_vars]  = target_encoder.transform(x_test[categorical_vars])
```

### Feature Selection {#lin-feature-selection}

`RFECV` uses cross-validation to keep only the features that pull their weight. It consistently keeps **`outcode` and `floorAreaSqM`** and drops `propertyType` — location and size are what matter.

```python
regressor = LinearRegression()
feature_selector = RFECV(regressor)
fit = feature_selector.fit(x_train, y_train)

x_train = x_train.loc[:, feature_selector.get_support()]
x_test  = x_test.loc[:, feature_selector.get_support()]
```

### Model Training {#lin-training}

```python
regressor = LinearRegression()
regressor.fit(x_train, y_train)
```

### Model Assessment {#lin-assessment}

```python
y_pred = regressor.predict(x_test)

r_squared = r2_score(y_test, y_pred)                       # ~0.75
cv = KFold(n_splits=4, shuffle=True)
cv_scores = cross_val_score(regressor, x_train, y_train, cv=cv, scoring="r2")   # ~0.74

n, k = x_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)   # ~0.75
```

An **R² of ~0.75** from a straight line is a strong baseline: with just location and floor area, the linear model already explains three-quarters of the variation in London sale prices. The coefficients point the expected way — bigger and more central means more expensive.

---

## 05. Random Forest

The forest uses the **same** prepared data as above, so the preparation steps are identical — repeated here for completeness, then the model itself differs.

### Data Import {#rf-data-import}

```python
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import TargetEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

model_data = pd.read_csv("kaggle_london_house_price_data.csv")
model_data["history_date"] = pd.to_datetime(model_data["history_date"], errors="coerce")
model_data = model_data[model_data["history_date"].dt.year == 2024]
model_data = model_data.drop_duplicates(subset=["fullAddress", "history_price"])
model_data["Price_per_SqM"] = model_data["history_price"] / model_data["floorAreaSqM"]

address_lookup = model_data["fullAddress"].copy()
model_data = model_data[["outcode", "floorAreaSqM", "propertyType", "history_price", "Price_per_SqM"]]
model_data = shuffle(model_data)
```

### Dealing with Missing Values {#rf-missing}

```python
model_data.dropna(how="any", inplace=True)
```

### Dealing with Outliers {#rf-outliers}

Same IQR rule on `Price_per_SqM` as the linear model.

```python
for column in ["Price_per_SqM"]:
    lower_quartile = model_data[column].quantile(0.25)
    upper_quartile = model_data[column].quantile(0.75)
    iqr_extended = (upper_quartile - lower_quartile) * 2
    outliers = model_data[(model_data[column] < lower_quartile - iqr_extended) |
                          (model_data[column] > upper_quartile + iqr_extended)].index
    model_data.drop(outliers, inplace=True)
```

### Splitting the Data {#rf-split}

```python
x = model_data.drop(["Price_per_SqM", "history_price"], axis=1)
y = model_data["history_price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### Target Encoding {#rf-encoding}

```python
categorical_vars = ["propertyType", "outcode"]
target_encoder = TargetEncoder(target_type="continuous")

x_train[categorical_vars] = target_encoder.fit_transform(x_train[categorical_vars], y_train)
x_test[categorical_vars]  = target_encoder.transform(x_test[categorical_vars])
```

*(Note: the forest keeps all three features — there's no `RFECV` step here, since tree ensembles are happy to ignore weak features on their own.)*

### Model Training {#rf-training}

```python
regressor = RandomForestRegressor(n_estimators=500)
regressor.fit(x_train, y_train)
```

### Model Assessment {#rf-assessment}

```python
y_pred = regressor.predict(x_test)
r_squared = r2_score(y_test, y_pred)   # ~0.76

cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, x_train, y_train, cv=cv, scoring="r2")   # ~0.73
```

I also read off both **feature importance** and **permutation importance** to see what the forest actually leans on:

```python
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(x_train.columns)
feature_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_summary.columns = ["input_variable", "feature_importance"]

result = permutation_importance(regressor, x_test, y_test, n_repeats=10, random_state=42)
```

Both tell the same story: **floor area and location do the heavy lifting**, with property type a minor contributor. The forest reaches **R² ~0.76** — statistically level with the linear model, which is exactly the feature-ceiling point from [01. Results](#01-results).

---

## 06. Detecting Undervalued Homes

A price predictor on its own is only mildly interesting. The actual product is what you do with its **residuals** — the gap between what a home *should* have sold for and what it *did*.

If the model predicts a fair price of £600k and the home sold for £300k, that 50% gap is worth a second look. I built exactly that on the test set, re-attaching each address **by index** so labels line up with rows even after the shuffle:

```python
undervaluation_threshold = 50   # flag homes that sold >= 50% below predicted

results = pd.DataFrame()
results["address"] = address_lookup.loc[x_test.index].values
results["actual_price"] = y_test.values
results["predicted_price"] = y_pred
results["undervalued_by_%"] = (results["predicted_price"] - results["actual_price"]) / results["predicted_price"] * 100

undervalued = results[results["undervalued_by_%"] >= undervaluation_threshold]
undervalued = undervalued.sort_values("undervalued_by_%", ascending=False)
```

Using `.loc[x_test.index]` (label-based) rather than `.iloc` is the detail that makes this trustworthy — I verified every one of the 1,253 test rows matched the raw file address-for-address, so the shuffle never scrambled the labels.

**The honest part -** When I looked at the *most* undervalued flags, they weren't bargains at all. The top hit was a Chelsea flat the model valued at £1.3m that "sold" for £60k. That isn't a steal — it's a **lease extension, shared-ownership share, or transfer between family**, not an open-market sale.

> The model only sees size, location and type. It doesn't know the further details. A residual flag is a **candidate to investigate, not a conclusion.** The tool's job is to hand over a short, ranked list.

---

## 07. Growth & Next Steps

A few concrete improvements would take this from a working prototype to something sturdier:

- **An upper-bound guard on the threshold**, so the shortlist surfaces *plausible* 10–30% discounts.
- **A fixed `random_state`** on the split, so the exact shortlist is reproducible run-to-run.
- **More signal to break the feature ceiling** — condition, lease years remaining, and transport links are the features most likely to push accuracy past ~0.75.

---
