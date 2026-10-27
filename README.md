# SmartCart Segmentation System

A customer segmentation project for **SmartCart**, an e-commerce platform. It uses unsupervised machine learning (clustering) to group customers by purchasing behaviour, engagement and loyalty, so marketing and retention can be personalised instead of generic.

## Problem
SmartCart treats all customers the same way. This leads to inefficient marketing, missed chances to retain high-value customers, and late detection of customers likely to churn. The goal is to discover hidden customer groups from historical data.

## Dataset
- 2,240 customer records, 22 attributes (demographics, spending, purchase channels, web activity, response)
- **The dataset is not included in this repository.** Place `smartcart_customers.csv` in the same folder as the notebook before running it.

## Approach (see `Smartcart.ipynb`)
1. **Missing values**: filled `Income` with the median
2. **Feature engineering**: created `Age`, `Customer_Tenure_Days`, `Total_spending`, `Total_Children`, and `Living_With` (Partner / Alone); grouped `Education` into Undergraduate / Graduate / Postgraduate
3. **Outlier removal**: dropped customers with Age ≥ 90 or Income ≥ 600,000 (2,240 → 2,236 rows)
4. **Correlation heatmap** and **one-hot encoding** of categorical columns
5. **Scaling** with StandardScaler
6. **PCA**: reduced to 3 components, which capture about 44% of the variance, for visualisation and clustering
7. **Choosing K**: Elbow method (with `kneed`) and Silhouette score, giving **K = 4**
8. **Clustering**: K-Means and Agglomerative (Ward linkage). Agglomerative showed better visual separation, so it was used for profiling
9. **Cluster profiling**: income vs spending plots and per-cluster averages

## Customer Segments Found

| Cluster | Avg income | Avg spending | Profile | Suggested action |
|---|---|---|---|---|
| 0 | ~39.7k | ~222 | Family shoppers: low/moderate income, more children, low response to campaigns | Coupons and family offers |
| 1 | ~72.8k | ~1,237 | High income and spending, live with a partner, buy more in-store and by catalog | Loyalty programs |
| 2 | ~37.0k | ~166 | Digital browsers: low spending, more children, most web visits | Heavy discounts during sales |
| 3 | ~70.7k | ~1,190 | High spenders living alone, older, best campaign response (~32%) | Premium and personalised services |

Clusters 1 and 3 are the premium customers, and clusters 0 and 2 are price-sensitive.

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (KMeans, Agglomerative Clustering, PCA), kneed, Jupyter Notebook

## How to Run
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
# add smartcart_customers.csv to this folder
jupyter notebook Smartcart.ipynb
```

## Files
- `Smartcart.ipynb`: preprocessing, PCA, clustering and cluster analysis

## Author
S Arunachalam
