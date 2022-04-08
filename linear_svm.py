import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

dx, dy = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)
dx_std = StandardScaler().fit_transform(dx)
dx_train, dx_test, dy_train, dy_test = train_test_split(
    dx_std, dy, test_size=0.2, random_state=0)

linear_svc = LinearSVC()
linear_svc.fit(dx_train, dy_train)
prediction = linear_svc.predict(dx_test)

print(linear_svc.score(dx_train, dy_train))
print(linear_svc.score(dx_test, dy_test))
