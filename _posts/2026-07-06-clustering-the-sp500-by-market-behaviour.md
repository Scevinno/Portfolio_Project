---
layout: post
title: Clustering the S&P 500 by Market Behaviour
image: "/img/posts/sp500_clustering.svg"
tags: [Machine Learning, Clustering, Python]
summary: "K-means groups ~420 S&P 500 companies into five distinct groups — on a 2025 dataset assembled via yfinance — to explore how stocks behave across the 11 defined sectors."
stack: "Python · pandas · scikit-learn · yfinance"
metrics:
  - value: "5"
    label: "cluster groups"
  - value: "~420"
    label: "2025 S&P companies"
---

Every S&P 500 company carries a sector label — Utilities, Information Technology, Health Care, and more. I collected data representing high-level detail on how each of 499 companies **actually behaved through 2025** — returns, volatility, valuation, profitability, growth — and let a K-means model group the 422 with complete records, with no knowledge of those labels. The question: do stocks tend to trend in a certain consistent way across their sector tags?

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Data Preparation](#04-data-preparation)
- [05. K-Means](#05-k-means)
  - [Feature Scaling](#kmeans-scaling)
  - [Choosing How Many Clusters](#kmeans-k)
  - [Model Training](#kmeans-training)
  - [Profiling the Clusters](#kmeans-profiling)
- [06. Do Stocks Follow Their Sector Labels?](#06-do-stocks-follow-their-sector-labels)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

Sector labels group companies by what they *sell*. This project is an exploration of what happens when you group them by how their stocks actually *behaved* — how they moved, earned and priced through 2025 — and then lay those groups over the sector map. The aim is to surface trends and groups — to see whether stocks tend to trend consistently within their sector, where they don't, and what interesting structure falls out along the way. The open-ended pattern exploration is exactly why unsupervised ML is used.

**Actions**

There was no ready-made table for this, so I built one: for every current S&P 500 constituent I assembled ten 2025-anchored metrics — price behaviour computed from a full year of daily prices, fundamentals from the four quarterly statements ending within 2025 — into one row per company. On that table I ran an unsupervised **K-means** pipeline — min-max scaling, WCSS search for the number of clusters, a reproducible seed — then profiled each cluster in real units and mapped its sector make-up to see what patterns had emerged.

**Applications**

Grouping stocks by behaviour is a great tool on any equity or wealth management platform. A portfolio manager can use the clusters as a concentration check — five holdings from five different sectors that all sit in the momentum cluster are one behavioural bet — and a research team can flag companies drifting away from their sector's typical behaviour as candidates for a closer look. The same pipeline fits fund research too, where style buckets like "defensive income" or "steady compounders" are exactly the tools model portfolios are assembled from.

**Growth & Next Steps**

The scaling choice is the first thing to revisit — min-max compresses heavy-tailed ratios like P/E and lets market cap's enormous range carve the nine biggest companies into a cluster of their own. Beyond that: a leaner set of input variables, and comparing each cluster's metrics against its sector's averages to investigate the outliers inside each sector.

---

## 01. Results

The model finds **five behavioural groups** — three findings stand out:

**Some sectors have behavioural consistency.** **87% of Utilities**, **85% of Real Estate** and **75% of Consumer Staples** companies land in the same cluster — the Defensive Income group that aligns exactly with the conservative reputation of those sectors. Their label genuinely tells you how they trade.

**The momentum growth cluster leans heavily towards tech.** More than half of its 24 companies carry the Information Technology label — Oracle, Palantir, AMD and Micron among them — a concentration most likely associated with 2025's growth in AI. Other sectors appear in the group too, but the tech tilt is the pattern that stands out.

**The remaining eight sectors are much more mixed.** Outside the defensive trio, behaviour is far less uniform — each sector's companies spread across several clusters, with Information Technology and Financials the most evenly split. An interesting next step is to explore the companies whose behaviour doesn't match the standard expectations for their particular sector.

| Cluster | Character | Companies | Return | Div. yield | Beta | Rev. growth | P/E ratio | Profit margin |
|---|---|---|---|---|---|---|---|---|
| 0 | Defensive income | 132 | +8% | 3.6% | 0.6 | +6% | 27× | 18% |
| 1 | Mega cap companies | 9 | +33% | 0.5% | 1.3 | +24% | 38× | 33% |
| 2 | Cyclicals | 112 | +10% | 1.0% | 1.3 | +8% | 43× | 12% |
| 3 | Momentum growth | 24 | +76% | 0.4% | 2.0 | +29% | 81× | 21% |
| 4 | Steady compounders | 145 | +6% | 1.1% | 0.6 | +8% | 74× | 14% |

---

## 02. Model Overview

**K-means** is an unsupervised algorithm: it doesn't require any training data. It works geometrically — each company becomes a point in ten-dimensional space, the algorithm drops k centre points among them, assigns every company to its nearest centre, moves each centre to the average position of the companies assigned to it, and repeats until nothing moves. The algorithm is reliant on being told how many groups to look for — so the WCSS elbow search in the K-Means section is how to arrive at a sensible number rather than inventing one.

---

## 03. Data Overview

Price metrics come from 2025's ~250 trading days (with the last 2024 close as baseline); fundamentals are trailing-twelve-month figures summed from the four quarterly statements ending within 2025, falling back to the nearest full fiscal year for off-cycle reporters. Four constituents without full 2025 history (2026 additions and spin-offs) were dropped, leaving **499 companies**.

| Variable Name | Variable Type | Description |
|---|---|---|
| ticker / company | Identifier | Ticker symbol and company name — dropped before modelling |
| sector | Label (held out) | One of the 11 GICS sectors — never shown to the model, used only to evaluate the clusters afterwards |
| annual_return_pct | Behavioural | Compounded 2025 price return |
| dividend_yield_pct | Behavioural | 2025 dividends paid ÷ start-of-year price |
| volatility_pct | Behavioural | Annualised standard deviation of daily returns |
| beta | Behavioural | Sensitivity to S&P 500 index moves across 2025 |
| pe_ratio | Valuation | Dec-2025 close ÷ trailing diluted EPS (blank if EPS ≤ 0) |
| roe_pct | Profitability | Return on equity, trailing twelve months |
| profit_margin_pct | Profitability | Net margin, trailing twelve months |
| eps_growth_pct | Growth | EPS growth, 2025 vs 2024 |
| revenue_growth_pct | Growth | Revenue growth, 2025 vs 2024 |
| market_cap_b | Scale | Market capitalisation in $ billions |

One deliberate choice: **missing values stay missing**. A company with negative earnings has no meaningful P/E. Gaps are handled at the modelling stage instead, where dropping them is an explicit, visible decision.

---

## 04. Data Preparation

K-means has no concept of a label, an identifier, or a text column — it just measures distances between rows. So preparation means two things here: deal with the gaps, and remove everything that isn't a behavioural measurement.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

data_for_clustering = pd.read_csv("sp500_2025_metrics.csv")

# rows with any missing metric are dropped — 499 down to 422
data_for_clustering.isna().sum()
data_for_clustering.dropna(how="any", inplace=True)

# identifiers and the sector label are held out of the feature set
data_for_clustering_scaled = data_for_clustering.drop(["ticker", "company", "fundamentals_basis", "sector"], axis=1)
```

The **sector column is deliberately held out**. The whole point is to see what groups emerge *without* it — it comes back at the end as the answer key.

---

## 05. K-Means

### Feature Scaling {#kmeans-scaling}

Market cap runs into the thousands of billions while beta lives between 0 and 3 — unscaled, every distance the model measures would be a market-cap comparison with noise attached. Min-max normalisation puts each of the ten metrics on the same 0–1 footing.

```python
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering_scaled),
                                          columns=data_for_clustering_scaled.columns)
```

### Choosing How Many Clusters {#kmeans-k}

K-means needs to be told how many groups to find. The standard tool is the **elbow method**: fit the model at every k from 1 to 11, record the Within-Cluster Sum of Squares (how tightly each cluster hugs its centre), and look for the point where adding another cluster stops buying much tightness. 

```python
k_values = list(range(1, 12))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values, wcss_list)
plt.title("Cluster Sum of Squares")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```

The curve bends gradually rather than snapping at one obvious point — common on real financial data, where group boundaries are soft. I chose **k = 5**: past five, each extra cluster shaved little off the WCSS, and the five groups that emerge are distinct enough to describe in plain English.

### Model Training {#kmeans-training}

```python
kmeans = KMeans(n_clusters=5, n_init=50, random_state=50)
kmeans.fit(data_for_clustering_scaled)

# attach each company's cluster back onto the readable data
data_for_clustering["cluster"] = kmeans.labels_
data_for_clustering["cluster"].value_counts()
```

Two arguments here do the reliability work. K-means starts from randomly placed centres, so a single unseeded run can land in a different local solution every time — early runs of this model reshuffled companies between clusters on every execution. `n_init=50` makes each fit try 50 different starting positions and keep only the best; with that in place the same five groups re-emerge under *any* seed, which is the real test that the structure lives in the data rather than in the randomness. `random_state` then pins the run so every number in this write-up reproduces exactly.

### Profiling the Clusters {#kmeans-profiling}

The labels go back onto the **unscaled** table, which matters for this step: profiling in real units. Averaging each metric per cluster turns five anonymous group numbers into five recognisable investor archetypes — the table in [01. Results](#01-results):

```python
cluster_summary = data_for_clustering.groupby("cluster")[["annual_return_pct", "dividend_yield_pct",
    "volatility_pct", "beta", "pe_ratio", "roe_pct", "profit_margin_pct",
    "eps_growth_pct", "revenue_growth_pct", "market_cap_b"]].mean().reset_index()
```

Reading the profiles is where the model earns its keep. A 3.6% average yield with a 0.6 beta is unmistakably a defensive income group; a 63% volatility with a 2.0 beta and +76% returns is unmistakably the momentum crowd. The model was never told these categories exist — it found them in the numbers.

---

## 06. Do Stocks Follow Their Sector Labels?

Now the held-out sector column comes back. Cross-tabulating sector against cluster — normalised within each sector — shows exactly how much of each sector shares one behaviour:

```python
sector_mix = pd.crosstab(data_for_clustering["sector"], data_for_clustering["cluster"], normalize="index")

ax = sector_mix.plot(kind="barh", stacked=True, figsize=(12, 6), width=0.85,
                     edgecolor="white", linewidth=1)
for bars in ax.containers:
    ax.bar_label(bars, labels=[f"{v:.0%}" if v >= 0.08 else "" for v in bars.datavalues],
                 label_type="center", fontsize=10, color="white")

plt.title("Cluster Composition Within Each Sector")
plt.xlabel("Share of sector's companies")
plt.ylabel("")
plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
plt.box(False)
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

Each sector's bar splits into the behavioural clusters its companies actually landed in — sorted from the most tribal sector to the most scattered:

![Cluster composition within each sector — stacked bar chart]({{ "/img/posts/sp500_sector_clusters.png" | relative_url }})

For a portfolio builder the practical takeaway is direct: holding five different tech names may still be one concentrated behavioural bet — or five genuinely different ones.

---

## 07. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Rethink the min-max scaling.** It puts every metric on a 0–1 range, but it is defenceless against extremes. A P/E and market cap's enormous range are two to mention. Log-transforming the heavy-tailed columns, or switching to a robust scaler, would let both metrics contribute a gradient instead of an outlier flag.
- **Reduce the number of variables.** Ten metrics all get an equal vote, but some of them overlap — volatility and beta tell related stories, as do the two growth columns. A leaner, less-correlated feature set could tell a much simpler and cleaner story.
- **Compare clusters to sector averages.** Each cluster's profile can be set against the average metrics of each sector — showing which companies sit far from their own sector's norm. That turns the clustering into a practical screen for investigating outliers within each individual sector, one sector at a time.

---
