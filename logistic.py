import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dx, dy = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)
dx_std = StandardScaler().fit_transform(dx)
dx_train, dx_test, dy_train, dy_test = train_test_split(
    dx_std, dy, test_size=0.2, random_state=0)
logistic = LogisticRegression()
logistic.fit(dx_train, dy_train)
prediction = logistic.predict(dx_test)

print(logistic.score(dx_train, dy_train))
print(logistic.score(dx_test, dy_test))
