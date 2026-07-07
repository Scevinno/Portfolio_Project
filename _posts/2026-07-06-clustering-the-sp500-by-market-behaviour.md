---
layout: post
title: Clustering the S&P 500 by Market Behaviour
image: "/img/posts/sp500_clustering.svg"
tags: [Machine Learning, Clustering, Python]
summary: "K-means groups 422 S&P 500 companies into five behavioural tribes — on a 2025 dataset I assembled from scratch — to test whether stocks really behave the way their sector labels say they should."
stack: "Python · pandas · scikit-learn · yfinance"
metrics:
  - value: "5"
    label: "behavioural groups"
  - value: "422"
    label: "companies clustered"
---

Every S&P 500 company carries a sector label — Utilities, Information Technology, Health Care — assigned by a classification committee, not by the market. I built a dataset of how each of 499 companies **actually behaved through 2025** — returns, volatility, valuation, profitability, growth — and let a K-means model group the 422 with complete records, with no knowledge of those labels. The question: do stocks behave like their sector says they should?

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Data Overview](#02-data-overview)
- [03. Data Preparation](#03-data-preparation)
- [04. K-Means](#04-k-means)
  - [Feature Scaling](#kmeans-scaling)
  - [Choosing How Many Clusters](#kmeans-k)
  - [Model Training](#kmeans-training)
  - [Profiling the Clusters](#kmeans-profiling)
- [05. Do Stocks Follow Their Sector Labels?](#05-do-stocks-follow-their-sector-labels)
- [06. Growth & Next Steps](#06-growth--next-steps)

---

## 00. Project Overview

**Context**

Sector labels drive real decisions — index funds, sector ETFs, portfolio diversification rules all lean on them. But a label is a statement about what a company *sells*, not how its stock *behaves*. If two "diversified" holdings actually move, earn and price identically, the diversification is on paper only. Clustering by measured behaviour puts the labels to the test.

**Actions**

There was no ready-made table for this, so I built one: for every current S&P 500 constituent I assembled ten 2025-anchored metrics — price behaviour computed from a full year of daily prices, fundamentals from the four quarterly statements ending within 2025 — into one row per company. On that table I ran an unsupervised **K-means** pipeline: min-max scaling, an elbow (WCSS) search for the number of clusters, a seeded 50-restart fit so the groups are stable and reproducible, then profiling each cluster and mapping its sector make-up.

**Growth & Next Steps**

Heavy-tailed ratios like P/E compress under min-max scaling, so a log-transform or robust scaler is the next modelling step, followed by a silhouette check on the cluster count. The more ambitious extension is clustering on daily return correlation — grouping stocks by how they move *together*, not just by their summary statistics.

---

## 01. Results

The model finds **five behavioural groups** — and they only partly respect the sector map. Three findings stand out:

**Some sectors are real behavioural tribes.** **87% of Utilities**, **85% of Real Estate** and **75% of Consumer Staples** companies land in the same cluster — a low-beta, high-dividend, low-volatility group that behaves exactly like the defensive reputation of those sectors. Their label genuinely tells you how they trade.

**But across the whole index, the labels are only half-right.** Just **55% of companies** sit in their sector's most common behavioural group. Information Technology is the sharpest example: its biggest slice (43%) doesn't behave like "tech" at all — it trades alongside the economy-linked cyclicals — while the rest scatters across all four other groups. Two stocks with the same tech label can sit in completely different behavioural worlds.

**The market has a momentum tribe that ignores sector lines.** A cluster of 24 companies averaged a **+76% return in 2025** at more than double the market's volatility and a beta of 2.0. Tesla, Oracle, Palantir, AMD and Micron sit in it — but so do power utilities riding the AI build-out (Constellation, Vistra), crypto-adjacent finance (Coinbase, Robinhood) and data-centre industrials (Vertiv). Behaviour, not sector, is what these stocks share.

| Cluster | Companies | Return | Div. yield | Volatility | Beta | Character |
|---|---|---|---|---|---|---|
| Defensive income | 132 | +8% | 3.6% | 26% | 0.6 | Utilities, real estate & staples heartland |
| Steady compounders | 145 | +6% | 1.1% | 27% | 0.6 | The index's calm, reinvesting core |
| Cyclical workhorses | 112 | +10% | 1.0% | 41% | 1.3 | Economy-linked industrials, semis & banks |
| Momentum growth | 24 | +76% | 0.4% | 63% | 2.0 | The 2025 rocket ships |
| Mega-cap giants | 9 | +33% | 0.5% | 37% | 1.3 | The $1.6–4.5T platform businesses |

---

## 02. Data Overview

The dataset is the part of this project I'm proudest of — every number in it is computed from dated source data, anchored to **calendar 2025** so no metric leaks information from a different period. Price metrics come from 2025's ~250 trading days (with the last 2024 close as baseline); fundamentals are trailing-twelve-month figures summed from the four quarterly statements ending within 2025, falling back to the nearest full fiscal year for off-cycle reporters. Four constituents without full 2025 history (2026 additions and spin-offs) were dropped, leaving **499 companies**.

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

One deliberate choice: **missing values stay missing**. A company with negative earnings has no meaningful P/E — writing one in would be inventing data. Gaps are handled at the modelling stage instead, where dropping them is an explicit, visible decision.

---

## 03. Data Preparation

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

## 04. K-Means

### Feature Scaling {#kmeans-scaling}

Scaling is not optional for K-means. Market cap runs into the thousands of billions while beta lives between 0 and 3 — unscaled, every distance the model measures would be a market-cap comparison with noise attached. Min-max normalisation puts each of the ten metrics on the same 0–1 footing, giving every metric an equal vote.

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

The curve bends gradually rather than snapping at one obvious point — common on real financial data, where group boundaries are soft. I chose **k = 5**: past five, each extra cluster shaved little off the WCSS, and the five groups that emerge are distinct enough to describe in plain English — which is its own kind of validation for an unsupervised model.

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

## 05. Do Stocks Follow Their Sector Labels?

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

The spectrum runs from tribal to scattered:

| Sector | Largest single cluster | Reading |
|---|---|---|
| Utilities | 87% | The label works — one shared behaviour |
| Real Estate | 85% | Same defensive block, different address |
| Consumer Staples | 75% | The classic defensive heartland |
| Health Care | 58% | Mostly steady compounders, with a fringe |
| Information Technology | 43% | Scattered across all five groups — and its biggest slice behaves like a cyclical, not like "tech" |
| Financials | 42% | The most divided label — banks ≠ exchanges ≠ insurers |

For a portfolio builder the practical takeaway is direct: holding five different tech names may still be one concentrated behavioural bet — or five genuinely different ones. The sector label can't tell you; the behaviour can.

---

## 06. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Tame the heavy tails.** A handful of true-but-extreme values (a P/E above 6,000, an ROE near 4,000%) compress everyone else into a sliver of the min-max scale, muting those metrics' influence — and they drag cluster *averages* around, which is why the profiling step deserves a median view too. Log-transforming the ratio columns — or switching to a robust scaler — would let them speak properly.
- **Validate the cluster count.** The elbow read was a judgement call; a silhouette-score sweep across k would put a number on it.
- **Deduplicate dual share classes.** Alphabet enters twice (GOOG and GOOGL) — one row per company is the honest count.
- **Cluster on co-movement.** Summary statistics describe each stock alone. Clustering on the correlation of daily returns would group stocks by how they move *together* — the definition of behaviour that matters most for diversification.

---
