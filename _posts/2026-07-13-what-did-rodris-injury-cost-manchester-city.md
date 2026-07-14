---
layout: post
title: Measuring the Causal Impact of Rodri's Injury on Manchester City
image: "/img/posts/rodri_causal_impact.svg"
tags: [Causal Inference, Time Series, Python]
summary: "A causal impact model builds the season Manchester City should have had if it weren't for Rodri's ACL injury — and measures the gap between the counterfactual and what actually happened."
stack: "Python · pandas · pycausalimpact"
metrics:
  - value: "−33%"
    label: "relative effect"
  - value: "~70"
    label: "matches analysed 2023–25"
  - value: "−0.81"
    label: "points per match"
---

Early in the 2024–25 season, Rodri — the midfielder in Manchester City — suffered a serious knee injury and was ruled out for the rest of the season. City had spent years as the dominant team in Premier League; without him they fell away almost immediately, and the season ended as their weakest in nearly a decade. This project asks the question every fan argued about — **how much of that collapse was one player?** — and answers it with a causal impact model: build the season City *should* have had, then measure the gap.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Data Preparation](#04-data-preparation)
- [05. Causal Impact](#05-causal-impact)
  - [Preparing the Series](#ci-series)
  - [Defining the Periods](#ci-periods)
  - [Fitting & Reading the Model](#ci-fitting)
- [06. The Cost in Points](#06-the-cost-in-points)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

A before-and-after comparison can show that City got worse after the injury — but worse *than what*? Comparing average points before and after isn't enough on its own, because plenty of other things change from one season to the next. To say what the injury itself cost, you need an estimate of how the season would have gone *without* it. That is what causal impact analysis builds: it learns what City's results looked like while Rodri was fit, then projects the season they would most likely have had if nothing had changed. The cost of the injury is the gap between that projected season and the one that actually happened.

**Actions**

I assembled a match-level dataset spanning one year either side of the injury date — every City league match, its points, and a control series built from six fixed clubs — then fitted a Bayesian structural time-series model (**CausalImpact**) with the injury as the intervention point, and read the effect off the gap between the observed and predicted seasons.

**Applications**

Quantifying what a single player's absence costs carries a real importance in football. Clubs weigh exactly this when setting transfer and wage budgets — a midfielder whose absence costs 0.8 points a game justifies a very different fee from one whose impact is less significant. The same measurement supports squad planning or performance departments making the case for load management with points rather than opinions.

**Growth & Next Steps**

The model judges City against a benchmark — currently the results of six other top clubs over the same period. Bookmaker odds would make a better benchmark: they exist for every match and already factor in how hard each game is expected to be. Beyond that, deeper match statistics would show *how* the points were lost — fewer goals scored, more conceded, or both — and running the same analysis on other clubs' long injuries would show whether an effect this size is unusual or normal.

---

## 01. Results

The model predicts that without the injury, City would have averaged **2.49 points per match** over the 31 games that followed it. They actually averaged **1.68**.

| | Average per match | Cumulative (31 matches) |
|---|---|---|
| Actual | 1.68 | 52 |
| Predicted without injury | 2.49 | 77 |
| 95% interval on the prediction | [1.98, 2.99] | [61, 93] |
| **Effect** | **−0.81** | **−25** |

**The gap is large and clearly separated from zero.** Even at the cautious end of the interval City lost around 9 points to the period after the injury; at the centre of the estimate it is 25 — roughly the distance between defending the title and finishing outside the top four.

**The defeats tell the same story from another angle.** City lost 3 of the 37 matches in the year before the injury and 9 of the 31 after it, with goals conceded per match rising from 0.9 to 1.2. A defensive midfielder's absence showing up first in the defensive numbers is the pattern you would expect if the injury were doing the damage.

**The effect is an average-level finding, not a match-level one.** Points in football arrive as 0, 1 or 3 and single matches are dominated by noise — the model makes no claim about which specific fixtures were dropped, only that the season ran persistently below its expected level.

---

## 02. Model Overview

**CausalImpact** answers a question an A/B test cannot: what would have happened without the event, when there is only one Manchester City and the season cannot be re-run. It fits a Bayesian structural time-series model to the pre-injury period, learning City's underlying level of performance and how it relates to a control series the injury does not affect. From the injury onward, the model projects that relationship forward to produce a counterfactual — the most likely path of the season had nothing changed — with an uncertainty band around it. The measured impact is simply the observed season minus the counterfactual, and because the model carries its uncertainty through, the result arrives with a credible interval rather than a bare number.

---

## 03. Data Overview

One row per Manchester City league match, spanning a year each side of the injury: **37 matches before** (22 September 2023 – 21 September 2024) and **31 after** (23 September 2024 – 19 May 2025). The injury match itself is excluded, and the post-period stops the day before Rodri's first appearance back, so neither window is contaminated by a partly-fit Rodri.

| Variable Name | Variable Type | Description |
|---|---|---|
| date / opponent / venue | Identifier | When, against whom, home or away |
| goals_for / goals_against | Outcome | Full-time score from City's perspective |
| result / points | Outcome | W/D/L and its 3/1/0 encoding — points is the modelled variable |
| rodri_played | Flag | Whether Rodri featured, verified match-by-match against his season logs |
| period | Window | pre or post |
| top6_avg_points | Control | Average points per match of six fixed clubs over the days since City's previous match |

Two details matter in the control column. The six clubs are **fixed** — Arsenal, Liverpool, Chelsea, Manchester United, Tottenham, Newcastle — regardless of league position, and their matches *against City* are excluded, because results in those fixtures are themselves affected by the injury. And the flag exposed an honest wrinkle in the pre-period: Rodri missed 7 of the 37 matches before the injury (suspensions, a rest, and a slow post-Euro start), so the "with Rodri" baseline is really a 30-of-37 baseline.

---

## 04. Data Preparation

The dataset was assembled from published season results: filter City's fixtures out of two seasons of Premier League results, convert each match to points from City's perspective, and label the two windows around the injury date. The control column aggregates the six clubs' results between consecutive City match dates, so each row carries a contemporaneous reading of how the league's other big sides were performing. Rodri's availability per match came from his published match logs rather than memory — which is exactly how the three August 2024 matches he sat out, and the mid-season rest between two Champions League legs, ended up correctly flagged.

---

## 05. Causal Impact

### Preparing the Series {#ci-series}

CausalImpact expects the measured series in the first column and the control alongside it. Matches are irregularly spaced in calendar time — congested weeks, international breaks — so the index is the match number, one step per match:

```python
from causalimpact import CausalImpact
import pandas as pd

causal_impact_df = pd.read_csv("rodri_city_matches.csv")

# index by match number
causal_impact_df.index = range(1, len(causal_impact_df) + 1)

# select columns
causal_impact_df = causal_impact_df[["points", "top6_avg_points"]]
```

### Defining the Periods {#ci-periods}

The injury falls between match 37 and match 38:

```python
pre_period = [1, 37]
post_period = [38, 68]
```

### Fitting & Reading the Model {#ci-fitting}

```python
ci = CausalImpact(causal_impact_df, pre_period, post_period)

ci.plot()
print(ci.summary())
print(ci.summary(output = "report"))
```

The summary produces the numbers in [01. Results](#01-results). One reading note: with an outcome that only takes the values 0, 1 and 3, the per-match panels of the plot are dominated by noise — the counterfactual is close to a flat line and individual matches scatter around it. The signal lives in the cumulative view, where thirty-one small gaps compound into something unmistakable.

---

## 06. The Cost in Points

The cumulative effect traces how the cost accumulated match by match — hovering near zero through October, then falling steadily as the winter fixtures arrived, with the shaded band marking the 95% interval:

![Cumulative points effect after the injury]({{ "/img/posts/rodri_cumulative_effect.png" | relative_url }})

The curve never recovers toward zero. Whatever was lost was not a short adjustment period — the season ran below its expected level from November to May.

---

## 07. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Strengthen the control series.** Club form is a serviceable control but a noisy one. Bookmaker odds are published for every match and encode fixture difficulty directly — converting them to expected points would give the model a control that tracks the measured series far more tightly.
- **Trace the mechanism with underlying metrics.** Points say what was lost; expected goals for and against would say how — whether the drop came from creating less, conceding more, or both.
- **Repeat the design for other absences.** The same pipeline applies to any long injury at any club. Running it across several cases would show whether a 0.8-points-per-match effect is exceptional or simply what losing a key player costs.

---
