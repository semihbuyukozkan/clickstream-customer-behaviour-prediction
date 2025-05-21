# clickstream-customer-behaviour-prediction
Research project on predicting customer purchases using e-commerce clickstream data. Includes EDA and models like Logistic Regression, Random Forest, XGBoost, LSTM, and S2L. SMOTE is used to handle class imbalance. Developed for a scientific research paper.
---

## 📁 Dataset

This project uses the **RetailRocket Recommender System Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

> ⚠️ **Before running any models, please download the dataset and place the `events.csv` file in the following directory: /dataset/events.csv

Only the `events.csv` file is required for this project.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/semihbuyukozkan/clickstream-customer-behaviour-prediction.git
cd clickstream-customer-behaviour-prediction

### 2. (Optional but recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # For windows: .venv\Scripts\activate

### 3. Install required dependencies
pip install -r requirements.txt

### 4. Download the dataset
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
and place only the `events.csv` file into the `dataset/` directory.

### 5. Generate features for machine learning models
python preprocessing.py

### 6. Run any model script (examples)
python LogisticRegression.py
python RandomForest.py
python XGBoost.py
python LSTM.py
python S2L.py



