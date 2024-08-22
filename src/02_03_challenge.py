import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap
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

# Sample the dataset to reduce processing time
X_train_sample = X_train.sample(1000, random_state=42)
X_test_sample = X_test.sample(1000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
y_test_sample = y_test.loc[X_test_sample.index]

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sample, y_train_sample)

# Make predictions
y_pred = model.predict(X_test_sample)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_sample, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_sample, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_sample, y_pred))

# SHAP Analysis: Using a smaller sample for SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

# If shap_values is a list, select the appropriate class (1 for positive class)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Visualize Feature Importance
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('output/02_03_challenge_ai_policy_shap_summary_sampled.png')