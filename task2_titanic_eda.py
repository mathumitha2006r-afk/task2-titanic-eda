"""
Prodigy InfoTech - Data Science Internship - Task 2
----------------------------------------------------
Objective: Perform data cleaning and exploratory data analysis (EDA) on the
Titanic dataset. Explore relationships between variables and identify
patterns and trends in the data.

HOW TO GET THE DATA:
1. Go to: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%202
   OR download "train.csv" from the Kaggle Titanic competition:
   https://www.kaggle.com/c/titanic/data
2. Save it in the SAME folder as this script and update CSV_PATH below if the
   filename is different.

Typical Titanic columns:
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare,
Cabin, Embarked
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# =======================================================================
# 1. LOAD THE DATA
# =======================================================================
CSV_PATH = "titanic.csv"  # <-- change this to your downloaded file's name

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("INITIAL DATA OVERVIEW")
print("=" * 70)
print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe(include="all"))

# =======================================================================
# 2. DATA CLEANING
# =======================================================================
print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)
print(df.isnull().sum())

df_clean = df.copy()

# --- Age: fill missing values with the median age (grouped by Pclass & Sex
#     gives a more realistic estimate than a single global median) ---
if "Age" in df_clean.columns:
    df_clean["Age"] = df_clean.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# --- Embarked: fill missing values with the mode (most common port) ---
if "Embarked" in df_clean.columns:
    df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# --- Fare: fill any missing values with the median fare ---
if "Fare" in df_clean.columns:
    df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].median())

# --- Cabin: mostly missing -> create a simpler "Has_Cabin" flag instead of
#     trying to impute the actual cabin number ---
if "Cabin" in df_clean.columns:
    df_clean["Has_Cabin"] = df_clean["Cabin"].notnull().astype(int)
    df_clean.drop(columns=["Cabin"], inplace=True)

# --- Drop columns that don't help with analysis ---
for col in ["Ticket", "Name", "PassengerId"]:
    if col in df_clean.columns:
        df_clean.drop(columns=[col], inplace=True)

# --- Remove exact duplicate rows, if any ---
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"\nRemoved {before - len(df_clean)} duplicate rows.")

# --- Feature engineering: Family size, IsAlone ---
if "SibSp" in df_clean.columns and "Parch" in df_clean.columns:
    df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
    df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

print("\n" + "=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)
print(df_clean.isnull().sum())

df_clean.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")

# =======================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =======================================================================

# --- 3.1 Survival counts ---
plt.figure(figsize=(6, 5))
sns.countplot(data=df_clean, x="Survived", hue="Survived", palette="Set2", legend=False)
plt.title("Survival Count (0 = Died, 1 = Survived)", fontweight="bold")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("01_survival_count.png", dpi=150)
plt.show()

# --- 3.2 Survival rate by sex ---
plt.figure(figsize=(6, 5))
sns.barplot(data=df_clean, x="Sex", y="Survived", hue="Sex", palette="Set1", legend=False)
plt.title("Survival Rate by Sex", fontweight="bold")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("02_survival_by_sex.png", dpi=150)
plt.show()

# --- 3.3 Survival rate by passenger class ---
plt.figure(figsize=(6, 5))
sns.barplot(data=df_clean, x="Pclass", y="Survived", hue="Pclass", palette="viridis", legend=False)
plt.title("Survival Rate by Passenger Class", fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("03_survival_by_class.png", dpi=150)
plt.show()

# --- 3.4 Age distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["Age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution of Passengers", fontweight="bold")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("04_age_distribution.png", dpi=150)
plt.show()

# --- 3.5 Age distribution split by survival ---
plt.figure(figsize=(8, 5))
sns.histplot(data=df_clean, x="Age", hue="Survived", bins=30, kde=True,
             palette="Set1", element="step")
plt.title("Age Distribution by Survival Status", fontweight="bold")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("05_age_by_survival.png", dpi=150)
plt.show()

# --- 3.6 Fare distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["Fare"], bins=40, kde=True, color="darkorange")
plt.title("Fare Distribution", fontweight="bold")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("06_fare_distribution.png", dpi=150)
plt.show()

# --- 3.7 Survival by embarkation port ---
if "Embarked" in df_clean.columns:
    plt.figure(figsize=(6, 5))
    sns.barplot(data=df_clean, x="Embarked", y="Survived", hue="Embarked",
                palette="pastel", legend=False)
    plt.title("Survival Rate by Port of Embarkation", fontweight="bold")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("07_survival_by_embarked.png", dpi=150)
    plt.show()

# --- 3.8 Family size vs survival ---
if "FamilySize" in df_clean.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_clean, x="FamilySize", y="Survived", hue="FamilySize",
                palette="coolwarm", legend=False)
    plt.title("Survival Rate by Family Size", fontweight="bold")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("08_survival_by_familysize.png", dpi=150)
    plt.show()

# --- 3.9 Correlation heatmap (numeric variables only) ---
plt.figure(figsize=(10, 8))
numeric_df = df_clean.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features", fontweight="bold")
plt.tight_layout()
plt.savefig("09_correlation_heatmap.png", dpi=150)
plt.show()

# --- 3.10 Pairplot of key variables ---
key_vars = [c for c in ["Survived", "Pclass", "Age", "Fare", "FamilySize"]
            if c in df_clean.columns]
sns.pairplot(df_clean[key_vars], hue="Survived", palette="Set1", diag_kind="kde")
plt.savefig("10_pairplot.png", dpi=150)
plt.show()

# =======================================================================
# 4. KEY INSIGHTS (printed to console)
# =======================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
overall_rate = df_clean["Survived"].mean()
print(f"Overall survival rate: {overall_rate:.1%}")

if "Sex" in df_clean.columns:
    print("\nSurvival rate by sex:")
    print(df_clean.groupby("Sex")["Survived"].mean())

if "Pclass" in df_clean.columns:
    print("\nSurvival rate by passenger class:")
    print(df_clean.groupby("Pclass")["Survived"].mean())

if "Embarked" in df_clean.columns:
    print("\nSurvival rate by embarkation port:")
    print(df_clean.groupby("Embarked")["Survived"].mean())

print("\nAll charts saved as PNG files in the current folder.")
print("Cleaned dataset saved as titanic_cleaned.csv")
