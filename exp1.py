import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
housing = fetch_california_housing()
data = pd.DataFrame(housing.data,
                    columns=housing.feature_names)
data["Price"] = housing.target
x = data[['AveRooms']].values
y = data['Price'].values

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    
    random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

w=0
b=0
learning_rate = 0.01
epochs = 1000

n=len(x_train_scaled)
cost_history=[]

for i in range(epochs):
    y_pred = w*x_train_scaled.flatten()+b
    dw = (1/n) * np.sum((y_pred - y_train) * x_train_scaled.flatten())
    db = (1/n) * np.sum(y_pred - y_train)
    w = w - learning_rate*dw
    b = b-learning_rate*db
    if i % 100 ==0:
         cost = (1/(2*n))*np.sum((y_pred - y_train)**2)
         cost_history.append(cost)
         print(f"Epoch{i}, Cost = {cost:.4f}")
       
y_pred_gd = w * x_test_scaled.flatten()+b

print("Gradient Descent")
print("------------------")
print("Weight:",w)
print("Bias:",b)
print("MSE:",mean_squared_error(y_test,y_pred_ne))
print("R2 Score:",r2_score(y_test,y_pred_ne))

x_train_ne = np.c_[np.ones((len(x_train),1)),x_train]
x_test_ne = np.c_[np.ones((len(x_test),1)),x_test]
theta = np.linalg.inv(x_train_ne.T @ x_train_ne) @ x_train_ne.T @ y_train
y_pred_ne = x_test_ne @ theta
print("\n Normal Equation")
print("------------------")
print("Intercept:",theta[0])
print("Slope:",theta[1])
print("MSE:",mean_squared_error(y_test,y_pred_ne))
print("R2 Score:",r2_score(y_test,y_pred_ne))

sort_axis = np.argsort(x_test.flatten())
x_sorted = x_test[sort_axis]
y_pred_ne_sorted = y_pred_ne[sort_axis]

index = np.argsort(x_test.flatten())
plt.figure(figsize=(8,5))
plt.scatter(x_test,y_test,color='blue',label='Actual Data')

plt.plot(
    x_test.flatten()[index],
    y_pred_ne[index],
    color='red',
    linewidth=2,
    label='Regression Line'
)
plt.title("Linear Regression")
plt.xlabel("Average Rooms")
plt.ylabel("Medium House Value")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(cost_history, color='navy',linewidth=2)
plt.title('Gradient Descent Cost Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost J (0)')
plt.grid(True , linestyle='--',alpha=0.6)
plt.show()

