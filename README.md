# Multi-Warehouse Replenishment and Fulfillment Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Forecasting](https://img.shields.io/badge/Forecasting-Demand%20Prediction-orange)
![Optimization](https://img.shields.io/badge/Optimization-Replenishment%20Planning-yellow)
![Supply Chain](https://img.shields.io/badge/Supply%20Chain-Fulfillment-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-red)
![Operations Research](https://img.shields.io/badge/OR-Decision%20Modeling-purple)

##  Project Overview

This project develops an end-to-end supply chain optimization pipeline that integrates:

- Demand Forecasting (Machine Learning)
- Censored Demand Adjustment
- Scenario-Based Demand Modeling
- Inventory Replenishment Optimization
- Service-Level-Constrained Decision Making

The objective is to support data-driven replenishment planning under uncertainty, balancing cost efficiency and inventory protection.

---

##  Problem Statement

In real-world supply chains, companies must decide:

- How much to replenish for each SKU at each warehouse
- While minimizing total operational cost
- And maintaining a desired service level under demand uncertainty

This project formulates the problem as:

> Demand Forecasting → Demand Adjustment → Scenario Modeling → Optimization under Constraints

---

##  Dataset

This project uses the M5 Forecasting dataset, which includes:

- Daily SKU-level sales data
- Store-level and item-level identifiers
- Calendar features (weekday, events)
- Price information

---

##  Methodology

### 1️⃣ Data Processing

- Convert raw data into long format
- Merge calendar and price features

### 2️⃣ Demand Forecasting

- Baseline Model: LightGBM
- Enhanced Model: Feature-enriched LightGBM

### 3️⃣ Censored Demand Identification

- Identify lost demand
- Estimate true demand

### 4️⃣ Scenario Modeling

- Low / Base / High demand scenarios
- Compute uncertainty and safety factors

### 5️⃣ Optimization

- Baseline: cost minimization
- Service-level: safety stock for high-risk SKUs

---

##  Key Results

| Metric | Baseline | Service-Level |
|------|--------|--------------|
| Replenishment | 387,222 | 399,770 |
| Ending Inventory | 0 | 12,548 |
| Total Cost | 468,995 | 487,067 |

---

##  How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

##  Author

Andrea Kang
