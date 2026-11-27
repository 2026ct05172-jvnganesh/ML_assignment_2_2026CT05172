# Airline Passenger Satisfaction – Classification with 5 ML Models

## 1. Problem statement
Predict whether an airline passenger is **satisfied** or **neutral/dissatisfied** from flight details, passenger profile and 14 service ratings. Five classifiers (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest) are trained on the same data and compared with Accuracy, AUC, Precision, Recall, F1 and MCC.

## 2. Dataset description
- **Source:** Kaggle – Airline Passenger Satisfaction (TJ Klein): https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- **Size:** 129,880 rows (train.csv 103,904 + test.csv 25,976 combined), 22 input features after dropping `Unnamed: 0` and `id`
- **Target:** `satisfaction` (satisfied = 1, neutral or dissatisfied = 0)
- **Feature types:** 4 categorical (Gender, Customer Type, Type of Travel, Class), 14 ordinal service ratings (0–5), 4 continuous (Age, Flight Distance, Departure Delay, Arrival Delay)
- **Class distribution:** 56.55 % neutral/dissatisfied vs 43.45 % satisfied (ratio about 1.3 : 1, mild imbalance)
- **Missing values:** 393 in `Arrival Delay in Minutes` (median-imputed). **Duplicates:** 0.
- **Preprocessing:** 80:20 stratified split with seed 42 done first; median imputation + StandardScaler on numeric columns and one-hot encoding on categorical columns, all fitted on the training set only (via a scikit-learn Pipeline).

- **Test data:** `test_data.csv` is the 20 % hold-out set (25,976 rows) saved *before* preprocessing, so it still contains the original raw values (including 64 missing `Arrival Delay in Minutes` values). The saved pipelines impute, scale and encode it themselves.

## 3. GitHub repository link
`(https://github.com/2026ct05172-jvnganesh/ML_assignment_2_2026CT05172/tree/main)`

## 4. Models used
Test set = 25,976 rows (20 %).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8741 | 0.9272 | 0.8699 | 0.8352 | 0.8522 | 0.7431 |
| Decision Tree | 0.9448 | 0.9870 | 0.9504 | 0.9211 | 0.9355 | 0.8877 |
| kNN | 0.9282 | 0.9703 | 0.9409 | 0.8907 | 0.9151 | 0.8539 |
| Naive Bayes | 0.8484 | 0.9159 | 0.8306 | 0.8179 | 0.8242 | 0.6911 |
| Random Forest (Ensemble) | 0.9623 | 0.9940 | 0.9712 | 0.9413 | 0.9560 | 0.9235 |

Model configurations: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=10`, `min_samples_leaf=5`), kNN (`k=5`), GaussianNB (defaults), Random Forest (`n_estimators=200`). Seed = 42.

### Metric visualisations
![Metrics](images/metrics_bar_charts.png)
![Grouped](images/metrics_grouped.png)
![ROC](images/roc_curves.png)

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Linear boundary misses interactions between ratings and travel type; clear baseline, fast and interpretable. |
| Decision Tree | Depth-limited tree captures non-linear rules, strong and explainable, but less stable than the ensemble. |
| kNN | Good after scaling, but lower recall than the tree models and slow at prediction time on about 104k rows. |
| Naive Bayes | Weakest: correlated service ratings violate the independence assumption. |
| Random Forest (Ensemble) | Best on all six metrics; averaging 200 trees reduces variance. Top features: online boarding, inflight wifi, class, type of travel. |
| **Overall winner** | **Random Forest** |

## 5. How to run
```bash
pip install -r requirements.txt
# download train.csv and test.csv from the Kaggle link above and place them in the model/ folder
cd model
jupyter notebook ML_Assignment_Airline_Satisfaction.ipynb   # Kernel > Restart & Run All
```
Running the notebook regenerates:
- `../test_data.csv` – the 20 % hold-out set
- `../images/*.png` – the charts used in this README
- `model/*.joblib` – the five trained pipelines (preprocessing + model)

## 6. Repository structure
```
requirements.txt
README.md
test_data.csv
images/    (metrics_bar_charts.png, metrics_grouped.png, roc_curves.png)
model/     (ML_Assignment_Airline_Satisfaction.ipynb)
```
