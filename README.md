# Task 2 - Data Cleaning & EDA on Titanic Dataset (Prodigy InfoTech Data Science Internship)

## Objective
Perform data cleaning and exploratory data analysis (EDA) on the Titanic
dataset. Explore relationships between variables and identify patterns and
trends in the data.

## Steps to run
1. Download the Titanic dataset CSV (train.csv) from either:
   - https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%202
   - https://www.kaggle.com/c/titanic/data
2. Place the downloaded CSV in this same folder.
3. Open `task2_titanic_eda.py` and set `CSV_PATH` to your file's name
   (e.g. "train.csv" or "titanic.csv").
4. Install requirements if needed:
   pip install pandas numpy matplotlib seaborn
5. Run the script:
   python task2_titanic_eda.py

## What the script does

### Data Cleaning
- Fills missing `Age` values using the median age grouped by Pclass & Sex
- Fills missing `Embarked` values with the most common port
- Fills missing `Fare` values with the median fare
- Converts `Cabin` into a simple `Has_Cabin` flag (mostly missing otherwise)
- Drops unhelpful columns (`Ticket`, `Name`, `PassengerId`)
- Removes duplicate rows
- Engineers new features: `FamilySize` and `IsAlone`
- Saves the cleaned dataset as `titanic_cleaned.csv`

### Exploratory Data Analysis (10 charts saved as PNGs)
1. Survival count
2. Survival rate by sex
3. Survival rate by passenger class
4. Age distribution
5. Age distribution split by survival
6. Fare distribution
7. Survival rate by port of embarkation
8. Survival rate by family size
9. Correlation heatmap of numeric features
10. Pairplot of key variables (Survived, Pclass, Age, Fare, FamilySize)

### Console output
Prints missing-value counts before/after cleaning, summary statistics, and
key survival-rate insights broken down by sex, class, and embarkation port.
