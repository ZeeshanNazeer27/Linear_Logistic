import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('Salary_dataset.csv')

# Step 2 :Check for missing values
print(df.isnull().sum())

# Drop the 'Numbers' column if it exists
if 'Numbers' in df.columns:
    df = df.drop('Numbers', axis=1)
data = df.dropna()
print(df.head())

# Step 4 : Define features and target variable
X = df[['YearsExperience']]  
y = df['Salary']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Step 7: Evaluate Linear Regression Model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Step 9 : Results Plot
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred_test, color='red', linewidth=1)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

sns.regplot(x=y_test, y=y_pred_test, line_kws={'color': 'red'})
plt.xlabel('Actual Salary')
plt.ylabel('Predicted')
plt.show()

# step 10 : Accuracy is not used in regression problems so mean_squared_error and other measures are used
print(f"Train MSE:{train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Train R2 Score: {train_r2:2f}")
print(f"Test R2 Score: {test_r2:.2f}")