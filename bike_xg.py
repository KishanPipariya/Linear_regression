import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb 
data = np.genfromtxt('Bike-Sharing-Dataset/hour.csv', delimiter=',', skip_header=1)
#X = data[:][:][2:14]
#2:14
X = data[:, 2:14]
y = data[:, 16:17]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.4, max_depth = 100, alpha = 10, n_estimators = 10) 
xg_reg.fit(X_train,y_train) 
y_pred = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

print("Training dataset:")
y_pred = xg_reg.predict(X_train)
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))

print("Testing dataset:")
y_pred = xg_reg.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print("RMSE: %f" % (rmse)) 
