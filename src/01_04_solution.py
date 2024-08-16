import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Preprocessing
df.dropna(inplace=True)  # Handle missing values by dropping them

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df.drop('class_>50K', axis=1)
y = df['class_>50K']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Fairness Analysis: Analyze performance across different demographic groups
def fairness_analysis(df, feature, target):
    group_metrics = {}
    groups = df[feature].unique()
    
    for group in groups:
        group_data = df[df[feature] == group]
        X_group = group_data.drop(target, axis=1)
        y_group = group_data[target]
        y_group_pred = model.predict(X_group)
        accuracy = accuracy_score(y_group, y_group_pred)
        group_metrics[group] = accuracy
    
    return group_metrics

# Analyze fairness across sex
sex_fairness = fairness_analysis(df, 'sex_Male', 'class_>50K')
print("\nFairness Analysis by Sex:")
print(sex_fairness)

# Analyze fairness across race
race_fairness = fairness_analysis(df, 'race_White', 'class_>50K')
print("\nFairness Analysis by Race:")
print(race_fairness)

# Visualization of fairness analysis
sns.barplot(x=list(sex_fairness.keys()), y=list(sex_fairness.values()))
plt.title('Fairness Analysis by Sex')
plt.xlabel('Sex')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('output/01_04_solution_fairness_analysis_by_sex.png')

sns.barplot(x=list(race_fairness.keys()), y=list(race_fairness.values()))
plt.title('Fairness Analysis by Race')
plt.xlabel('Race')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('output/01_04_solution_fairness_analysis_by_race.png')

# Mitigation: Propose and implement techniques to mitigate bias
# For example, equalizing the performance by using techniques like reweighting, oversampling, etc.
# Implementation of mitigation techniques will depend on the identified bias.