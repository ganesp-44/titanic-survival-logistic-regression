# Dataset Directory (`data/`)

This directory stores the faculty-supplied Titanic dataset (`titanic.csv`, 891 rows x 12 columns).

## Redistribution & Git Tracking Note
If your institution restricts public redistribution of course datasets on GitHub:
1. Keep `data/titanic.csv` locally on your machine for running `src/train_model.py` and `notebooks/titanic_logistic_regression.ipynb`.
2. Add `data/*.csv` and `data/*.xlsx` to `.gitignore` before pushing to a public repository.
3. Keep `models/titanic_logistic_model.joblib` committed so `app.py` can still run predictions on Streamlit Community Cloud without needing the raw dataset.
