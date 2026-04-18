import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (classification_report, precision_score, recall_score, 
                             f1_score, roc_auc_score, precision_recall_curve)
import joblib
import os

RANDOM_STATE = 42

def load_data_and_model():
    if os.path.exists('split_data.pkl'):
        return joblib.load('split_data.pkl')
    else:
        raise FileNotFoundError("split_data.pkl not found. Please run model_training.py first.")

def optimize_best_model():
    X_train_res, y_train_res, X_test, y_test = load_data_and_model()
    
    print("\n================================================================")
    print("STEP 7: OPTIMIZE XGBOOST")
    print("================================================================")
    
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'scale_pos_weight': [1, 5, 10]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    
    print("Starting RandomizedSearchCV (this may take a while)...")
    random_cv = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        cv=5,
        scoring='roc_auc',
        n_iter=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    random_cv.fit(X_train_res, y_train_res)
    
    print(f"\nBest parameters found: {random_cv.best_params_}")
    
    best_xgb = random_cv.best_estimator_
    
    # Re-evaluate tuned model
    y_pred = best_xgb.predict(X_test)
    y_prob = best_xgb.predict_proba(X_test)[:, 1]
    
    print("\n--- Tuned XGBoost Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(best_xgb.feature_importances_, index=X_test.columns)
    feat_importances.nlargest(15).sort_values().plot(kind='barh', color='skyblue')
    plt.title('Top 15 Feature Importances (XGBoost)')
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
    
    return best_xgb, X_test, y_test

def threshold_tuning(model, X_test, y_test):
    print("\n================================================================")
    print("STEP 8: THRESHOLD TUNING")
    print("================================================================")
    
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = [0.5, 0.4, 0.3, 0.2]
    
    metrics_list = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred_t)
        r = recall_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)
        
        print(f"\nThreshold: {t}")
        print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f1:.4f}")
        
        metrics_list.append({'Threshold': t, 'Precision': p, 'Recall': r, 'F1-Score': f1})
    
    # Plot P vs R vs Threshold
    precisions, recalls, thresholds_curve = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_curve, precisions[:-1], label="Precision", color="blue")
    plt.plot(thresholds_curve, recalls[:-1], label="Recall", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision vs Recall vs Threshold")
    plt.legend()
    plt.savefig('threshold_tradeoff.png')
    print("\nThreshold tradeoff plot saved as 'threshold_tradeoff.png'")
    
    best_t = 0.3 # Typical recommendation for fraud where recall is priority
    print(f"\nRecommended Threshold: {best_t}")
    print("Explanation: Lowering the threshold increases RECALL (catching more fraud), "
          "but decreases PRECISION (more false alarms for legit users). "
          "0.3 is a strong balance for high-risk detection.")

def save_and_verify(model):
    print("\n================================================================")
    print("STEP 9: SAVE MODEL")
    print("================================================================")
    
    joblib.dump(model, 'fraud_model.pkl')
    print("Model saved as 'fraud_model.pkl'")
    
    # Reload and verify
    reloaded_model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Sample prediction (first row of X_test)
    X_train_res, y_train_res, X_test, y_test = load_data_and_model()
    sample_input = X_test.iloc[[0]]
    
    orig_pred = model.predict_proba(sample_input)
    new_pred = reloaded_model.predict_proba(sample_input)
    
    print(f"\nVerification - Original Prediction: {orig_pred}")
    print(f"Verification - Reloaded Prediction: {new_pred}")
    
    if np.allclose(orig_pred, new_pred):
        print("Model verification SUCCESSFUL: Predictions match.")
    else:
        print("Model verification FAILED: Predictions do not match.")

if __name__ == "__main__":
    try:
        best_model, X_test, y_test = optimize_best_model()
        threshold_tuning(best_model, X_test, y_test)
        save_and_verify(best_model)
    except Exception as e:
        print(f"An error occurred: {e}")
