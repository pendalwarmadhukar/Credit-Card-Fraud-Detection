import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """
    Step 1: Setup & Data Loading
    Loads the Credit Card Fraud dataset from a reliable public mirror.
    """
    url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    print("================================================================")
    print("STEP 1: SETUP & DATA LOADING")
    print("================================================================")
    
    print(f"Loading dataset from: {url}...")
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to local if URL fails
        if os.path.exists('creditcard.csv'):
            df = pd.read_csv('creditcard.csv')
            print("Loaded from local creditcard.csv")
        else:
            raise FileNotFoundError("Dataset not found. Please provide creditcard.csv.")

    # Display dataset info
    print(f"\nShape of dataset: {df.shape}")
    print("\nData Types and Null Values:")
    print(df.info())
    
    # Class distribution
    class_counts = df['Class'].value_counts()
    class_pct = df['Class'].value_counts(normalize=True) * 100
    print("\nClass Distribution:")
    print(pd.concat([class_counts, class_pct], axis=1, keys=['Count', 'Percentage']))
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Plot Class Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Fraud vs Legitimate Transaction Counts')
    plt.xlabel('Class (0: Legitimate, 1: Fraud)')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    print("\nClass distribution plot saved as 'class_distribution.png'")
    
    return df

def perform_eda(df):
    """
    Step 2: Exploratory Data Analysis (EDA)
    """
    print("\n================================================================")
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("================================================================")
    
    # 1. Distribution of Amount for fraud vs legitimate
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['Class'] == 0]['Amount'], label='Legitimate', fill=True)
    sns.kdeplot(df[df['Class'] == 1]['Amount'], label='Fraud', fill=True)
    plt.title('Distribution of Transaction Amount by Class')
    plt.xlabel('Amount')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('amount_distribution.png')
    print("Amount distribution plot saved as 'amount_distribution.png'")

    # 2. Distribution of Time for fraud vs legitimate
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['Class'] == 0]['Time'], label='Legitimate', fill=True)
    sns.kdeplot(df[df['Class'] == 1]['Time'], label='Fraud', fill=True)
    plt.title('Distribution of Transaction Time by Class')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('time_distribution.png')
    print("Time distribution plot saved as 'time_distribution.png'")

    # 3. Correlation heatmap of top 10 features with 'Class'
    print("\nCalculating top 10 feature correlations with Class...")
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    top_10_features = correlations.index[1:11] # Exclude 'Class' itself
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_10_features.tolist() + ['Class']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap: Top 10 Features with Class')
    plt.savefig('correlation_heatmap.png')
    print("Correlation heatmap saved as 'correlation_heatmap.png'")

    # 4. Key statistics of Amount by class
    print("\nKey Statistics for 'Amount' by Class:")
    stats_amount = df.groupby('Class')['Amount'].describe()[['mean', 'std', 'min', 'max']]
    print(stats_amount)

if __name__ == "__main__":
    try:
        df = load_data()
        perform_eda(df)
        # Save raw dataframe for next step if needed, or we'll just reload in the next script
        df.to_pickle("processed_df_raw.pkl")
        print("\nEDA Completed. Data temporarily saved to 'processed_df_raw.pkl'")
    except Exception as e:
        print(f"An error occurred: {e}")
