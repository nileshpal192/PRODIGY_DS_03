🌳 Task-03 — Decision Tree Classifier for Customer Purchase Prediction

This repository contains Task-03 of my Data Science Internship at Prodigy InfoTech.
The goal of this task is to build a Decision Tree Classifier capable of predicting whether a customer will purchase a product or service based on their demographic and behavioral information.

🎯 Task Objective

The primary objective is to apply machine learning classification techniques and understand how decision trees function in predictive modeling.

Using a dataset such as the Bank Marketing Dataset from the UCI Machine Learning Repository, the classifier predicts the likelihood of customer purchase (Yes/No).

📁 Sample Dataset

A sample dataset for this task is available here:
🔗 Dataset Link: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203

(Datasets include attributes such as customer age, job category, marital status, account balance, communication type, campaign duration, and other related features.)

🧪 Steps Performed in This Task

⿡ Data Loading & Understanding

Loaded the dataset using Pandas

Examined structure using .head(), .info(), and .describe()

Identified input features (X) and the target variable (y)

⿢ Data Cleaning

Handled missing values

Encoded categorical variables (Label Encoding / One-Hot Encoding)

Normalized or scaled numerical features (if required)

⿣ Splitting Data

Divided the dataset into:

Training Set (80%)

Testing Set (20%)

Used the train_test_split() function from sklearn

⿤ Model Building — Decision Tree Classifier

Implemented DecisionTreeClassifier() from sklearn

Trained the model on training data

Generated predictions on the test dataset

⿥ Model Evaluation

Evaluated the model using:

Accuracy Score

Confusion Matrix

Classification Report

Optional: Visualized the Decision Tree for interpretability

📊 Expected Output

The final output of this task includes:

Prediction of whether a customer will make a purchase (Yes/No)

Performance metrics evaluating the classifier

Visual interpretation of the decision tree structure

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

