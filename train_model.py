import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
df = pd.read_csv('electricity_bill_dataset.csv')

df = df[['Units_Consumed', "AC_Usage_Hours_per_Day", "Heater_Usage_Hours_per_Day", "Electricity_Bill"]]

print(df.columns)
print(df.head())

# Scaling the data
scaler = StandardScaler()

x = df[['Units_Consumed', "AC_Usage_Hours_per_Day", "Heater_Usage_Hours_per_Day"]]
y = df['Electricity_Bill']

x_scaled = scaler.fit_transform(x)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

#Save the model and scaler
joblib.dump(model, 'Linear_electricity_bill_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
