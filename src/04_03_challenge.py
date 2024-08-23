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

# Subset Sampling: Use a smaller subset of the dataset to speed up processing
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
plt.savefig('output/04_03_challenge_decision_tree.png')

# Calculate feature importance and visualize it
# TODO: Create a bar chart to visualize the feature importance
# Hint: Use plt.barh() and plt.savefig('output/04_04_solution_feature_importance.png')

# Check for fairness across groups
# TODO: Implement a basic fairness check, e.g., comparing model accuracy for different groups
# Hint: You can start by printing the class distribution in the test set

# Challenge: Your task is to complete the code by visualizing feature importance, checking for fairness, and creating a comprehensive summary report.
print("Challenge: Your task is to complete the code by visualizing feature importance, checking for fairness, and creating a comprehensive summary report.")
