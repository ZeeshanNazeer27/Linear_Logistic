import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('Salary_dataset.csv')

# Step 2: Check for missing values
print(df.isnull().sum())

# Drop the 'Numbers' column if it exists
if 'Numbers' in df.columns:
    df = df.drop('Numbers', axis=1)
data = df.dropna()

# Step 4: Define features and target directly
X = df[['YearsExperience']] 
threshold = 45000
y = (df['Salary'] > threshold).astype(int)
print(df.head())

# Step 5: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Check if train-test split is correct
print('Training set class distribution:')
print(y_train.value_counts())
print('Testing set class distribution:')
print(y_test.value_counts())

# Step 6: Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_test = logistic_model.predict(X_test)
y_pred_train = logistic_model.predict(X_train)

# Step 7: Evaluate Logistic Regression Model
logistic_accuracy_train = accuracy_score(y_train, y_pred_train)
logistic_accuracy = accuracy_score(y_test, y_pred_test)
logistic_precision_train = precision_score(y_train, y_pred_train)
logistic_precision = precision_score(y_test, y_pred_test)
logistic_recall_train = recall_score(y_train, y_pred_train)
logistic_recall = recall_score(y_test, y_pred_test)
logistic_f1_train = f1_score(y_train, y_pred_train)
logistic_f1 = f1_score(y_test, y_pred_test)


# Step 8 :Confusion Matrix for Logistic Regression
conf_mat = confusion_matrix(y_test, y_pred_test)

# Step 9: Confusion Matrix Plot
plt.figure(figsize=(7, 4))
sns.heatmap(conf_mat, annot=True,  cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Step 10 :Accuracy For Logistic Model
print(f' Accuracy_train: {logistic_accuracy_train:.2f}')
print(f' Accuracy: {logistic_accuracy:.2f}')
print(f' Precision_train: {logistic_precision_train:.2f}')
print(f' Precision: {logistic_precision:.2f}')
print(f' Recall_train: {logistic_recall_train:.2f}')
print(f' Recall: {logistic_recall:.2f}')
print(f' F1 Score_train: {logistic_f1_train:.2f}')
print(f' F1 Score: {logistic_f1:.2f}')
