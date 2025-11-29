#prodigy infotech task 03

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import joblib


data = pd.read_csv("bank.csv", delimiter=";")
print("Dataset loaded successfully.\n")
print(data.head())
print("Dataset Shape:", data.shape)


print("\nMissing value summary:\n")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)
print("\nMissing values removed.\n")


label_encoder = LabelEncoder()

# Encode target variable
data['y'] = label_encoder.fit_transform(data['y'])

# Identify categorical columns
cat_columns = data.select_dtypes(include=['object']).columns
print("Categorical columns:", list(cat_columns))

# Encode each categorical column
for col in cat_columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

print("\nDataset after encoding:\n", data.head())
print("\nData types after encoding:\n", data.dtypes)

X = data.drop("y", axis=1)
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel training completed.\n")


y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)


plt.figure(figsize=(15, 8))
tree.plot_tree(model, filled=True, fontsize=7)
plt.title("Decision Tree Structure")
plt.show()


joblib.dump(model, "decision_tree_bank_model.pkl")
print("\nModel saved as 'decision_tree_bank_model.pkl'")
print("\nProcess completed successfully.")
