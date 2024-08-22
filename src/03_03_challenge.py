import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
adult_data = fetch_openml('adult', version=2, as_frame=True)
df = adult_data.frame

# Select key features for transparency
selected_features = ['education-num', 'hours-per-week']
X = df[selected_features]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO: Train a RandomForestClassifier model# 
# TODO: Visualize one of the decision trees in the forest# Hint: Use plt.figure(figsize=(20, 10)) and tree.plot_tree()
# # TODO: Use SHAP to explain a single prediction# Hint: Use shap.TreeExplainer and shap_values
# # TODO: Choose an instance to explain using SHAP
# # TODO: Save the visualizations and reports in the 'output' folder

print("Challenge: Your task is to complete the code by training the model, visualizing the decision tree, using SHAP for model interpretability, and creating a transparency report.")
