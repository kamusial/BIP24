import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC  #clasifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X, y = make_circles(50, factor=0.2, random_state=0, noise=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
#
# print('\nLogisticRegression')
# model = LogisticRegression()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))
#
# print('\nDecisionTreeClassifier')
# model = DecisionTreeClassifier(max_depth=2)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))
#
# print('\nKNeighborsClassifier')
# model = KNeighborsClassifier(9)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))
#
# print('\nSVC')
# model = SVC()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

# SVM for flowers
# df = pd.read_csv("iris.csv")
# species = {
#     "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
# }
# df["class_value"] = df["class"].map(species)
# df = df.query(" class_value!=2 ")
# plt.scatter(df.sepallength, df.sepalwidth, c=df.class_value)
#
# X = df[ ["sepallength","sepalwidth"] ]
# y = df.class_value
#
# model = SVC()
# model.fit(X, y)
# vector = model.support_vectors_
# print(vector)
# plt.scatter(vector[:, 0], vector[:, 1], c='r' )
# plt.show()
#
