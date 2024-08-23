import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Preprocessing Steps:
# Handling missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Select features for human-centric AI design
selected_features = ['age', 'education-num', 'hours-per-week']
X = df[selected_features]
y = df['class_>50K']

# Use a smaller subset of the dataset to speed up processing
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, random_state=42)  # Use only 10% of the data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

# Train the model with fewer estimators to reduce computation time
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Generate predictions and evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize one of the decision trees in the forest
plt.figure(figsize=(20, 10))
tree.plot_tree(model.estimators_[0], feature_names=selected_features, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('output/04_04_solution_decision_tree.png')

# Calculate feature importance and visualize it
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(selected_features, feature_importances)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig('output/04_04_solution_feature_importance.png')

# Check for fairness across groups
# Here we could add analysis to compare model performance across different demographic groups
# For simplicity, let's just print the class distribution in the test set
class_distribution = y_test.value_counts(normalize=True)
print("Class Distribution in Test Set:")
print(class_distribution)

# Example of creating a summary report
with open('output/04_04_solution_human_centric_report.txt', 'w') as report_file:
    report_file.write("Human-Centric AI Design Report\n")
    report_file.write("==============================\n")
    report_file.write(f"Model Accuracy: {accuracy_score(y_test, y_pred)}\n")
    report_file.write(f"Feature Importances: {feature_importances}\n")
    report_file.write(f"Class Distribution in Test Set: {class_distribution.to_dict()}\n")
    report_file.write("Please refer to the decision tree and feature importance visualizations for detailed insights.\n")