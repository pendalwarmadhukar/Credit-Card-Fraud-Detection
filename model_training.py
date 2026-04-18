import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score)
import joblib
import os

# Set seed for reproducibility
RANDOM_STATE = 42

def load_preprocessed_data():
    """Load the data from the previous step or mirror."""
    if os.path.exists("processed_df_raw.pkl"):
        return pd.read_pickle("processed_df_raw.pkl")
    else:
        url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
        print(f"Loading dataset from: {url}...")
        return pd.read_csv(url)

def run_training_pipeline():
    df = load_preprocessed_data()
    
    print("\n================================================================")
    print("STEP 3: PREPROCESSING")
    print("================================================================")
    
    # 1. Scale 'Amount' and 'Time'
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # 2. Drop original unscaled columns
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # 3. Define X and y
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    # Save scalers for production use
    joblib.dump(scaler_amount, 'scaler_amount.pkl')
    joblib.dump(scaler_time, 'scaler_time.pkl')
    print("Scalers saved as 'scaler_amount.pkl' and 'scaler_time.pkl'")

    print("\n================================================================")
    print("STEP 4: HANDLE CLASS IMBALANCE WITH SMOTE")
    print("================================================================")
    
    print(f"Class distribution before SMOTE: \n{y_train.value_counts()}")
    
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"Class distribution after SMOTE: \n{y_train_res.value_counts()}")

    print("\n================================================================")
    print("STEP 5: TRAIN 3 MODELS")
    print("================================================================")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }
    
    results = {}
    curves_data = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predict on ORIGINAL (non-SMOTE) test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"--- {name} Evaluation ---")
        print(classification_report(y_test, y_pred))
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc_score:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Store metrics
        results[name] = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC-ROC": auc_score,
            "CM": cm
        }
        
        # Store data for curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr, "tpr": tpr, 
            "prec": precision_vals, "rec": recall_vals
        }

    print("\n================================================================")
    print("STEP 6: EVALUATION & VISUALIZATION")
    print("================================================================")
    
    # 1. Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, data in curves_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {results[name]['AUC-ROC']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig('roc_comparison.png')
    
    # 2. Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for name, data in curves_data.items():
        plt.plot(data['rec'], data['prec'], label=f"{name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.savefig('pr_comparison.png')
    
    # 3. Side-by-side Confusion Matrix heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, (name, metrics) in enumerate(results.items()):
        sns.heatmap(metrics['CM'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.savefig('cm_comparison.png')
    
    # 4. Summary comparison table
    comparison_df = pd.DataFrame(results).T[['Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
    print("\nModel Comparison Table:")
    print(comparison_df)
    
    # 5. Best Model Conclusion
    best_model_name = comparison_df['Recall'].idxmax()
    print(f"\nBEST MODEL: {best_model_name}")
    print("Reason: Priority is catching fraud (Class 1), so we focus on high RECALL for the fraud class.")
    
    # Save best model temporarily
    joblib.dump(models[best_model_name], 'temp_best_model.pkl')
    # Save X_test, y_test for tuning script
    joblib.dump((X_train_res, y_train_res, X_test, y_test), 'split_data.pkl')
    
    print("\nTraining Phase Completed. Models and visualizations saved.")

if __name__ == "__main__":
    run_training_pipeline()
