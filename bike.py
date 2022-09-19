import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt('Bike-Sharing-Dataset\\hour.csv', delimiter=',', skip_header=1)
#X = data[:][:][2:14]
#2:14
X = data[:, 2:14]
y = data[:, 16:17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Naive Linear Regresion using linear features")
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)
polynomial_regression.fit(X_train, y_train)

print("Linear Regression using non-linear features")

print("Training dataset:")
y_pred = polynomial_regression.predict(X_train)
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))

print("Testing dataset:")
y_pred = polynomial_regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

polynomial_regression = make_pipeline(
    StandardScaler(with_mean=False),
    PolynomialFeatures(degree=5, include_bias=False),
    Ridge(alpha=1000),
    
)
print("Regularized linear regression:")
polynomial_regression.fit(X_train, y_train)
print("Training dataset:")
y_pred = polynomial_regression.predict(X_train)
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))

print("Testing dataset:")
y_pred = polynomial_regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %f" % r2_score(y_test, y_pred))
