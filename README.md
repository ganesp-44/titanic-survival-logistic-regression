# Titanic Passenger Survival Prediction Using Logistic Regression
**Fundamentals of Data Science (FDS) — Case Study 3 (Batch 3)**

A complete, reproducible, leakage-free machine learning project that investigates passenger survival aboard the RMS *Titanic* using an L2-regularized **Logistic Regression** classifier built with `scikit-learn`, accompanied by exploratory data analysis (EDA), statistical odds-ratio interpretation, an interactive **Streamlit** web application, an executed **Jupyter Notebook**, and a formal **Academic PDF Report**.

## 1. Repository Structure

```text
titanic-survival-logistic-regression/
├── README.md                                  # Project documentation, setup, deployment & reproducibility log
├── requirements.txt                           # Pinned Python package dependencies (Python 3.11.2)
├── app.py                                     # Streamlit web application for interactive survival prediction
├── src/
│   └── train_model.py                         # Standalone training, cross_validate & serialization script
├── models/
│   └── titanic_logistic_model.joblib          # Trained scikit-learn Pipeline (ColumnTransformer + LogisticRegression)
├── notebooks/
│   └── titanic_logistic_regression.ipynb      # Executed 14-section Jupyter Notebook with code, outputs & plots
├── reports/
│   └── titanic_survival_report.pdf            # 16-section Academic PDF Report with figures & interpretations
└── data/
    ├── README.md                              # Dataset documentation & local storage notes
    └── titanic.csv                            # Faculty-supplied Titanic dataset (891 rows x 12 columns)
```

## 2. Software Environment & Reproducibility Specifications

To guarantee identical numerical results across standalone scripts, Jupyter notebooks, and external `scikit-learn` environments, the project is locked to the following verified software versions and configuration:
- **Python:** `3.11.2`
- **Core Scientific Stack:** `numpy==2.1.3`, `pandas==2.2.3`, `scipy==1.15.2`, `scikit-learn==1.5.2`, `joblib==1.4.2`, `matplotlib==3.10.1`, `seaborn==0.13.2`, `openpyxl==3.1.5`
- **Partitioning & Cross-Validation:**
  - Train/Test Split: `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` ($n_{\text{train}} = 712, n_{\text{test}} = 179$)
  - 5-Fold Cross-Validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` evaluated via a single shared `cross_validate(pipeline, X_train, y_train, cv=skf, scoring=("accuracy", "roc_auc", "f1"), return_train_score=False)` call in both `src/train_model.py` and `notebooks/titanic_logistic_regression.ipynb`.

## 3. Requirements Checklist Mapping

| Faculty Requirement (`Logistic Regression.docx` & `CASE STUDIES-FDS.pdf`) | Satisfied In | Verification Summary |
| :--- | :--- | :--- |
| **1a. Load dataset & perform EDA** | `titanic_logistic_regression.ipynb` (Sec. 4–6), `titanic_survival_report.pdf` (Sec. 6, 8) | Audited both `titanic.csv` and `titanic (1).xlsx` ($N = 891, p = 12$); resolved `Ticket` mixed-type discrepancy. |
| **1b. Examine features, data types & summary statistics** | `titanic_logistic_regression.ipynb` (Sec. 5), `titanic_survival_report.pdf` (Sec. 6, 8) | Full numerical (`mean`, `std`, quartiles) and categorical (`unique`, `top`, `freq`) summaries and missingness audit. |
| **1c. Visualizations (histograms, box plots, group bar charts, correlation heatmap)** | `titanic_logistic_regression.ipynb` (Sec. 6), `titanic_survival_report.pdf` (Sec. 8, 12, Figs. 1–5) | Figures 1–5 include descriptive titles, axis labels, and academic interpretations. |
| **2a. Handle missing values (imputation)** | `train_model.py`, `titanic_logistic_regression.ipynb` (Sec. 7–9), `titanic_survival_report.pdf` (Sec. 9) | Training-only `SimpleImputer(strategy='median')` for `Age` ($19.87\%$ missing) and `SimpleImputer(strategy='most_frequent')` for `Embarked` ($0.22\%$ missing); `Cabin` ($77.10\%$ missing) excluded. |
| **2b. Encode categorical variables** | `train_model.py`, `titanic_logistic_regression.ipynb` (Sec. 7–9), `titanic_survival_report.pdf` (Sec. 9) | `OneHotEncoder(drop='first', sparse_output=False)` on `Sex` (ref: `female`) and `Embarked` (ref: `C`), inside `ColumnTransformer`. |
| **3a–b. Build & train Logistic Regression model** | `train_model.py`, `titanic_logistic_regression.ipynb` (Sec. 8–9), `titanic_survival_report.pdf` (Sec. 10) | Stratified 80/20 split (`random_state=42`), `StandardScaler` + L2 `LogisticRegression(C=1.0, solver='lbfgs')` + 5-fold Stratified CV (`cross_validate`). |
| **4a. Evaluate Accuracy, Precision, Recall, F1-score, ROC-AUC & plot ROC curve** | `train_model.py`, `titanic_logistic_regression.ipynb` (Sec. 10–11), `titanic_survival_report.pdf` (Sec. 11–12, Figs. 6–7) | Held-out test ($n=179$): Accuracy $= 80.45\%$, Precision $= 79.31\%$, Recall $= 66.67\%$, F1 $= 0.7244$, ROC-AUC $= 0.8433$. |
| **5a–b. Interpret coefficients & feature significance** | `titanic_logistic_regression.ipynb` (Sec. 12), `titanic_survival_report.pdf` (Sec. 13, Fig. 8) | Log-odds coefficients ($\beta_j$), Odds Ratios ($e^{\beta_j}$), 95% CIs, Wald $z$-statistics, and $p$-values. |
| **6. Streamlit Deployment (`app.py`)** | `app.py`, `models/titanic_logistic_model.joblib`, `requirements.txt` | Interactive passenger form, input validation, predicted class & survival probability bar, and expandable model summary. |
| **FDS PDF Report Guidelines (Title page, methodology, statistical findings, explanations)** | `reports/titanic_survival_report.pdf` | 16-section academic report with structured cover page, 8 captioned figures, and full statistical interpretations. |

## 4. Key Quantitative Findings (Synchronized Across All Deliverables)

- **Dataset Audit (`titanic.csv` vs. `titanic (1).xlsx`):** Both files contain **891 rows and 12 columns** with identical missing-value counts (`Cabin`: 687 [77.10%], `Age`: 177 [19.87%], `Embarked`: 2 [0.22%]). In `titanic (1).xlsx`, 661 purely numeric `Ticket` values are stored as integers (`int`) and 230 alphanumeric tickets as strings (`str`), whereas `titanic.csv` stores all 891 `Ticket` values uniformly as strings.
- **5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on Training Set, $n = 712$):**
  - **CV Accuracy:** $0.7949 \pm 0.0273$ ($79.49\% \pm 2.73\%$) — Per-fold scores: `[0.7972, 0.7972, 0.8028, 0.7465, 0.8310]`
  - **CV ROC-AUC:** $0.8549 \pm 0.0212$ ($85.49\% \pm 2.12\%$) — Per-fold scores: `[0.8569, 0.8268, 0.8762, 0.8351, 0.8796]`
  - **CV F1-Score:** $0.7246 \pm 0.0335$ — Per-fold scores: `[0.7339, 0.7071, 0.7358, 0.6727, 0.7736]`

### Training Set ($n = 712$) vs. Held-Out Test Set ($n = 179$) Performance

| Evaluation Metric | Formula | Training Set ($n=712$) | Held-Out Test Set ($n=179$) |
| :--- | :--- | :---: | :---: |
| **Accuracy** | $(\text{TP}+\text{TN})/N$ | $0.8076$ ($80.76\%$) | **0.8045** (**80.45%**) |
| **Precision (`Survived = 1`)** | $\text{TP}/(\text{TP}+\text{FP})$ | $0.7698$ ($76.98\%$) | **0.7931** (**79.31%**) |
| **Recall / Sensitivity (`Survived = 1`)** | $\text{TP}/(\text{TP}+\text{FN})$ | $0.7106$ ($71.06\%$) | **0.6667** (**66.67%**) |
| **Specificity / TNR (`Survived = 0`)** | $\text{TN}/(\text{TN}+\text{FP})$ | $0.8679$ ($86.79\%$) | **0.8909** (**89.09%**) |
| **F1-Score (`Survived = 1`)** | $2\text{PR}/(\text{P}+\text{R})$ | $0.7390$ | **0.7244** |
| **ROC-AUC Score** | Area under ROC curve | $0.8594$ | **0.8433** |
| **Confusion Matrix** | $(\text{TN}, \text{FP}, \text{FN}, \text{TP})$ | $\text{TN}=381, \text{FP}=58, \text{FN}=79, \text{TP}=194$ | $\text{TN}=98, \text{FP}=12, \text{FN}=23, \text{TP}=46$ |

### Fitted Logistic Regression Coefficients ($\beta_0 = +1.2568, \text{SE} = 0.2524$)

| Transformed Feature | Scale / Reference | Coefficient ($\beta_j$) | Std. Error | Wald $z$ | $p$-value | Odds Ratio ($e^{\beta_j}$) | Interpretation (Holding Other Features Constant) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Sex_male` | Binary (Ref: `female`) | $-2.5600$ | $0.2132$ | $-12.01$ | $<0.0001$ | $0.0773$ | Male survival odds are $92.27\%$ lower than female odds ($12.94\times$ higher odds for females). |
| `Pclass` | Standardized ($\mu=2.31, \sigma=0.83$) | $-0.9158$ | $0.1321$ | $-6.93$ | $<0.0001$ | $0.4002$ | $+1\text{ SD}$ toward 3rd class lowers survival odds by $59.98\%$. |
| `Age` | Standardized ($\mu=29.56, \sigma=13.02$) | $-0.5007$ | $0.1101$ | $-4.55$ | $<0.0001$ | $0.6061$ | $+1\text{ SD}$ ($+13.02\text{ yrs}$) lowers survival odds by $39.39\%$. |
| `Embarked_S` | Binary (Ref: `Cherbourg C`) | $-0.3841$ | $0.2466$ | $-1.56$ | $0.1192$ | $0.6810$ | Lower point odds than Cherbourg, but not statistically significant at $\alpha=0.05$. |
| `Embarked_Q` | Binary (Ref: `Cherbourg C`) | $+0.2891$ | $0.3957$ | $+0.73$ | $0.4650$ | $1.3353$ | Not statistically significant ($p = 0.4650$). |
| `SibSp` | Standardized ($\mu=0.49, \sigma=1.06$) | $-0.2573$ | $0.1250$ | $-2.06$ | $0.0396$ | $0.7731$ | $+1\text{ SD}$ in siblings/spouses lowers survival odds by $22.69\%$ ($p < 0.05$). |
| `Fare` | Standardized ($\mu=31.82, \sigma=48.03$) | $+0.1019$ | $0.1358$ | $+0.75$ | $0.4532$ | $1.1072$ | Positive direction, but collinear with `Pclass` ($r = -0.55$); $p = 0.4532$. |
| `Parch` | Standardized ($\mu=0.39, \sigma=0.84$) | $-0.0596$ | $0.1035$ | $-0.58$ | $0.5643$ | $0.9421$ | Negligible independent linear effect ($p = 0.5643$). |

## 5. Local Setup and Execution Instructions

### Step 1: Clone or Extract the Repository and Create a Virtual Environment
```bash
cd titanic-survival-logistic-regression
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train the Model and Verify Evaluation Metrics
```bash
python3 src/train_model.py
```

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook notebooks/titanic_logistic_regression.ipynb
```

### Step 4: Launch the Streamlit Web Application Locally
```bash
streamlit run app.py
```

## 6. Online Deployment Guide (Streamlit Community Cloud via GitHub)

1. **Initialize and Push to GitHub:**
   ```bash
   git init
   git branch -M main
   git add README.md requirements.txt app.py src/ models/ notebooks/ reports/ data/
   git commit -m "Complete FDS Case Study 3: Titanic Survival Prediction using Logistic Regression"
   git remote add origin https://github.com/<your-username>/titanic-survival-logistic-regression.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
   - Click **Create app** -> select your repository `titanic-survival-logistic-regression`, branch `main`, and main file path `app.py`.
   - Click **Deploy!**
