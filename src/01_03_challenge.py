import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Brief Overview of the Dataset
print("Dataset Overview:")
print("Columns in the dataset:", df.columns)
print("First few rows of the dataset:")
print(df.head())

# Explain the Dataset
print("\nDataset Description:")
print("The 'adult' dataset, also known as the 'Census Income' dataset, contains information about individuals from the 1994 US Census.")
print("It is used to predict whether an individual earns more than $50,000/year based on features such as age, education, and work hours.")

# Instructions for the Challenge
print("\nChallenge: Ethical Dilemma Analysis")
print("1. **Dataset Understanding:**")
print("   - Understand the structure of the dataset, including features and target variable.")
print("   - Key features include age, education level, capital gains/losses, hours worked per week, and demographic information.")
print("   - Target variable indicates whether an individual's income is greater than $50,000/year.")
print("\n2. **Preprocessing:**")
print("   - Handle missing values if necessary.")
print("   - Encode categorical variables (e.g., sex, race) for machine learning models.")
print("\n3. **Model Training and Evaluation:**")
print("   - Train a machine learning model (e.g., RandomForestClassifier) to predict the target variable.")
print("   - Evaluate the model's performance using metrics such as accuracy, confusion matrix, and classification report.")
print("\n4. **Fairness Analysis:**")
print("   - Analyze the model's performance across different demographic groups (e.g., sex, race).")
print("   - Calculate metrics like demographic parity to assess if the model's predictions are fair across these groups.")
print("   - Visualize the results to identify any potential biases in the model's predictions.")
print("\n5. **Ethical Considerations:**")
print("   - Reflect on the ethical implications of the model's predictions and fairness analysis.")
print("   - Consider how biases in the data or model could impact real-world decisions and outcomes.")

# Plot sample distribution for reference
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=df)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/01_03_challenge_distribution_income.png')
plt.show()

# Additional plots for understanding the data distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/01_03_challenge_age_distribution.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='education', data=df, order=df['education'].value_counts().index)
plt.title('Education Level Distribution')
plt.xlabel('Count')
plt.ylabel('Education Level')
plt.tight_layout()
plt.savefig('output/01_03_challenge_education_distribution.png')
plt.show()

# Your task:
# 1. Analyze fairness across different demographic groups (e.g., by sex or race).
# 2. Investigate potential bias in the model's predictions.
# 3. Propose and implement techniques to mitigate any identified biases.