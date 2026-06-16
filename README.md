# Student-Mark-Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)
study_hours = np.random.uniform(1.0, 10.0, 100)
exam_marks = 10 + 8.5 * study_hours + np.random.normal(0, 5, 100)
exam_marks = np.clip(exam_marks, 0, 100)
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Exam_Marks': exam_marks
})
print("--- First 5 Rows of the Dataset ---")
print(df.head())
print()
X = df[['Study_Hours']]
y = df['Exam_Marks']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Training Complete.")
print(f"Intercept (c): {model.intercept_:.2f}")
print(f"Coefficient/Slope (m): {model.coef_[0]:.2f}")
print()
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")
print()
custom_hours = pd.DataFrame({'Study_Hours': [7.5]})
predicted_mark = model.predict(custom_hours)
print(
    f"Prediction Example: A student studying for 7.5 hours "
    f"is predicted to score: {predicted_mark[0]:.2f}/100"
)
print()
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Marks')
sorted_indices = X_test['Study_Hours'].argsort()
X_test_sorted = X_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]
plt.plot(
    X_test_sorted,
    y_pred_sorted,
    color='red',
    linewidth=2,
    label='Regression Line'
)
plt.title('Study Hours vs Exam Marks (Test Set)')
plt.xlabel('Study Hours')
plt.ylabel('Exam Marks')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
