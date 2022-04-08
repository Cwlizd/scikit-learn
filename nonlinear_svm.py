import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

dx, dy = make_moons(n_samples=500, noise=0.15, random_state=0)
dx_std = StandardScaler().fit_transform(dx)
dx_train, dx_test, dy_train, dy_test = train_test_split(
    dx_std, dy, test_size=0.2, random_state=0)


linear_svm = LinearSVC()
linear_svm.fit(dx_train, dy_train)
prediction = linear_svm.predict(dx_test)

nonlinear_svm = SVC()
nonlinear_svm.fit(dx_train, dy_train)
prediction = nonlinear_svm.predict(dx_test)

print(linear_svm.score(dx_train, dy_train))
print(linear_svm.score(dx_test, dy_test))

print(nonlinear_svm.score(dx_train, dy_train))
print(nonlinear_svm.score(dx_test, dy_test))


plt.scatter(dx.T[0], dx.T[1], c=dy, cmap="Dark2")
plt.savefig("non_vs_linear_svm")
plt.show()
