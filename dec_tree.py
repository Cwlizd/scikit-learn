import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dx, dy = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    dx, dy, test_size=0.2, random_state=0)

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
prediciton = dtree.predict(x_test)

print(dtree.score(x_train, y_train))
print(dtree.score(x_test, y_test))
