import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_raw_data(n=891):
    """Generates a synthetic dataset structured exactly like the Kaggle Titanic data."""
    df = pd.DataFrame({
        'survived': np.random.choice([0, 1], size=n, p=[0.61, 0.39]),
        'pclass': np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], size=n, p=[0.65, 0.35]),
        'age': np.random.normal(loc=29, scale=14, size=n),
        'sibsp': np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.68, 0.23, 0.04, 0.02, 0.02, 0.01]),
        'parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.76, 0.13, 0.09, 0.01, 0.005, 0.002, 0.003]),
        'fare': np.random.exponential(scale=32, size=n),
        'embarked': np.random.choice(['S', 'C', 'Q'], size=n, p=[0.72, 0.20, 0.08]),
        'class': np.random.choice(['First', 'Second', 'Third'], size=n, p=[0.24, 0.21, 0.55]),
        'who': np.random.choice(['man', 'woman', 'child'], size=n, p=[0.60, 0.32, 0.08]),
        'adult_male': np.random.choice([True, False], size=n, p=[0.60, 0.40]),
        'deck': np.random.choice([np.nan, 'C', 'B', 'D', 'E', 'A', 'F', 'G'], size=n, p=[0.77, 0.05, 0.05, 0.04, 0.04, 0.02, 0.02, 0.01]),
        'embark_town': np.random.choice(['Southampton', 'Cherbourg', 'Queenstown'], size=n, p=[0.72, 0.20, 0.08]),
        'alive': np.random.choice(['no', 'yes'], size=n, p=[0.61, 0.39]),
        'alone': np.random.choice([True, False], size=n, p=[0.60, 0.40])
    })
    
    # Introduce random missing values into specific columns
    df.loc[df.sample(frac=0.20).index, 'age'] = np.nan
    df.loc[df.sample(frac=0.005).index, 'embarked'] = np.nan
    df.loc[df.sample(frac=0.005).index, 'embark_town'] = np.nan
    return df

def clean_data(df):
    """Executes the data cleaning pipeline."""
    print("--- 1. Starting Data Cleaning Pipeline ---")
    print(f"Initial missing values:\n{df.isnull().sum()}\n")
    
    # Handle the 'deck' column string vs NaN artifacts if any
    df['deck'] = df['deck'].replace('nan', np.nan)
    
    # Impute missing values for 'age' using the median age of respective sex and passenger class subgroups
    df['age'] = df.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
    
    # Impute missing categorical entries with the dominant mode
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
    
    # Drop highly sparse and redundant columns
    df_cleaned = df.drop(columns=['deck', 'alive', 'class'])
    
    print(f"Missing values after cleaning pipeline:\n{df_cleaned.isnull().sum()}")
    return df_cleaned

def run_eda(df):
    """Performs Exploratory Data Analysis and saves summary charts."""
    print("\n--- 2. Starting Exploratory Data Analysis (EDA) ---")
    
    # Statistical summaries
    print("\nSurvival Rate by Gender:")
    print(df.groupby('sex')['survived'].mean())
    
    print("\nSurvival Rate by Ticket Class:")
    print(df.groupby('pclass')['survived'].mean())
    
    # Set plotting theme
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Survival Rate by Gender
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x='sex', y='survived', palette='pastel', errorbar=None, ax=ax)
    ax.set_title('Survival Rate by Gender')
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Gender')
    plt.tight_layout()
    plt.savefig('survival_by_gender.png')
    plt.close()
    
    # Plot 2: Survival Rate by Passenger Class
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x='pclass', y='survived', palette='muted', errorbar=None, ax=ax)
    ax.set_title('Survival Rate by Passenger Class')
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Passenger Class (Pclass)')
    plt.tight_layout()
    plt.savefig('survival_by_class.png')
    plt.close()
    
    # Plot 3: Age Distribution split by Survival Status
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='survived', kde=True, multiple='stack', palette='Set2', bins=25, ax=ax)
    ax.set_title('Age Distribution: Survivors vs Non-Survivors')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('age_distribution_by_survival.png')
    plt.close()
    
    # Plot 4: Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("\nAll EDA plots ('survival_by_gender.png', 'survival_by_class.png', "
          "'age_distribution_by_survival.png', 'correlation_matrix.png') saved successfully.")

if __name__ == "__main__":
    # Simulate fetching or loading data
    raw_data = generate_raw_data()
    
    # Clean the dataset
    cleaned_data = clean_data(raw_data)
    
    # Save clean dataset to disk
    cleaned_data.to_csv('cleaned_titanic_data.csv', index=False)
    print("\nCleaned records written to 'cleaned_titanic_data.csv'")
    
    # Execute visualization and trend analysis
    run_eda(cleaned_data)