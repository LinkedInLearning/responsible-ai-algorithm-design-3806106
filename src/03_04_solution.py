import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import shap
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Handling missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Select key features for transparency
selected_features = ['education-num', 'hours-per-week']
X = df[selected_features]
y = df['class_>50K']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Visualize one of the decision trees in the forest
plt.figure(figsize=(20, 10))
tree.plot_tree(model.estimators_[0], feature_names=selected_features, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('output/03_04_solution_decision_tree.png')

# Evaluate model accuracy
model_accuracy = accuracy_score(y_test, model.predict(X_test))

# Generate a basic transparency report with feature names
with open('output/03_04_solution_transparency_report.txt', 'w') as report_file:
    report_file.write("Model Transparency Report\n")
    report_file.write("=========================\n")
    report_file.write(f"Model Accuracy: {model_accuracy}\n")
    report_file.write("Feature Importances:\n")
    for feature, importance in zip(selected_features, model.feature_importances_):
        report_file.write(f"- {feature}: {importance}\n")

print("Report saved to output/03_04_solution_transparency_report.txt")

# SHAP explanation for a single prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Choose an instance to explain (e.g., first instance in the test set)
instance_index = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][instance_index], X_test.iloc[instance_index], feature_names=selected_features, matplotlib=True)
plt.title("SHAP Force Plot for First Instance")
plt.savefig('output/03_04_solution_shap_force_plot.png')

print("SHAP force plot saved to output/03_04_solution_shap_force_plot.png")