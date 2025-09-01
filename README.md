# Loan Prediction using Machine Learning  

## 📌 Project Overview  
This project predicts **loan approval status** based on applicant information (income, education, employment, credit history, etc.). It uses a **machine learning pipeline** to preprocess data, handle class imbalance, train models, and evaluate performance.  

The dataset is a common benchmark in classification tasks and is ideal for demonstrating **data cleaning, feature engineering, and model building skills**.  

---

## 📂 Dataset  
- **Source:** [Loan Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)  
- **Description:** Contains demographic and financial details of loan applicants such as:  
  - Gender, Marital Status, Education, Employment Type  
  - Applicant & Coapplicant Income  
  - Loan Amount and Loan Term  
  - Credit History and Property Area  
- **Target:** `Loan_Status` (Y = Approved, N = Not Approved)  

---

## 📂 Project Workflow  
1. **Data Loading & Exploration**  
   - Loaded dataset into pandas DataFrame.  
   - Explored missing values, categorical vs. numerical features, and class imbalance.  

2. **Data Preprocessing**  
   - Handled missing values (median for numeric, most frequent for categorical).  
   - Encoded categorical features (Label Encoding / One-Hot Encoding).  
   - Checked for outliers (IQR method).  
   - Created engineered features such as **Total Income** and binary flags.  

3. **Class Imbalance Handling**  
   - Compared different strategies:  
     - `class_weight="balanced"` in Random Forest.  
     - **SMOTE** and **SMOTE + Tomek Links** hybrid resampling.  

4. **Model Training**  
   - Random Forest Classifier (tuned).  
   - Compared with other potential algorithms (future scope: XGBoost, LightGBM).  

5. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
   - Emphasis on improving recall for the minority class (loan denials).  
   - Threshold tuning and evaluation of trade-offs using ROC/PR curves.  

---

## 📊 Results  
- **Random Forest (tuned)** achieved:  
  - Accuracy: ~83%  
  - ROC-AUC: ~0.80  
  - Precision/Recall balance better for approvals than denials (reflecting real dataset difficulty).  

- **Insights:**  
  - Credit history is the strongest predictor.  
  - Applicant income shows heavy skew (log transform helps).  
  - Class imbalance (2:1) makes denials harder to predict.  

---

---
## 📌 Future Improvements

- Try gradient boosting models (XGBoost, LightGBM, CatBoost).

- Perform hyperparameter tuning with GridSearch/Optuna.

- Deploy the model using Streamlit or Flask for real-time predictions. (MAYBE)

## 🚀 How to Run  
1. Clone this repo:
   ```bash
   git clone https://github.com/yaman-77/loan-prediction.git
   cd loan-prediction
