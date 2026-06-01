# PRODIGY_DS_02

task
Perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset to explore relationships between variables and identify patterns in survival rates.

Dataset
Titanic passenger data — sourced from Kaggle (891 passengers, 12 columns)

Key Findings
Women survived at 74% vs only 19% of men — "women and children first" was real and reflected in the data
1st class passengers had a 63% survival rate vs just 24% for 3rd class — wealth significantly influenced survival chances
Passengers aged 20-30 had the highest casualty count — young male adults were expected to stay back, disproving the assumption that younger = safer
Children under 10 had a relatively high survival rate compared to adults
Data Cleaning
Filled 177 missing Age values with median age
Dropped Cabin column (687 missing values — too sparse to use)
Dropped 2 rows with missing Embarked values
Charts
Bar chart: Survival rate by gender
Bar chart: Survival rate by passenger class
Histogram: Age distribution of survivors vs non-survivors
Tools Used
Python (Pandas, Matplotlib)
GitHub
