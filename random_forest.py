import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dx, dy = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    dx, dy, test_size=0.2, random_state=0)
random_f = RandomForestClassifier()

random_f.fit(x_train, y_train)
prediction = random_f.predict(x_test)

print(random_f.score(x_train, y_train))
print(random_f.score(x_test, y_test))
