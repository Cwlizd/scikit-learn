import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dx, dy = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)
dx_std = StandardScaler().fit_transform(dx)

plt.scatter(dx_std.T[0], dx_std.T[1], c=dy, cmap="Dark2")
plt.title("KNN-test")
plt.savefig("KNN-test.jpg")
plt.show()


dx_train, dx_test, dy_train, dy_test = train_test_split(
    dx_std, dy, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(dx_train, dy_train)
prediction = knn.predict(dx_test)
print(knn.score(dx_train, dy_train))
print(knn.score(dx_test, dy_test))
