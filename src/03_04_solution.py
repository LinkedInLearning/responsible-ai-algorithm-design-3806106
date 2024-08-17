import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Selecting features for interaction analysis
selected_features = ['education-num', 'hours-per-week']
X = df[selected_features]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Visualize one of the decision trees in the forest
plt.figure(figsize=(20, 10))
tree.plot_tree(model.estimators_[0], feature_names=selected_features, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('output/03_04_solution_decision_tree.png')

# Explain a single prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Choose an instance to explain
instance_index = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][instance_index,:], X_test.iloc[instance_index,:], feature_names=selected_features, show=False)
plt.savefig('output/03_04_solution_local_explanation.png')

# Example of creating a summary report (this would be more detailed in a real-world scenario)
# This could be exported as a PDF, HTML report, or integrated into a dashboard.
with open('output/03_04_solution_transparency_report.txt', 'w') as report_file:
    report_file.write("Model Transparency Report\n")
    report_file.write("=========================\n")
    report_file.write(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test))}\n")
    report_file.write(f"Feature Importances: {model.feature_importances_}\n")
    report_file.write("See attached SHAP summary plots and local explanations for detailed insights.\n")
