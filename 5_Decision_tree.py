import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

print(df.columns)
# sns.heatmap(    df.iloc[ : , : 4 ].corr(), annot=True          )
# plt.show()
#
# X = df[ ['sepallength', 'sepalwidth' ]    ]   #first leaf - sepal
# # X = df[ ['petallength', 'petalwidth' ]    ]    #second leaf - petal
# # X = df.iloc[: , :4] # both leafs together
# y = df.class_value
#
# from sklearn.tree import DecisionTreeClassifier
#
# model = DecisionTreeClassifier(max_depth=7, random_state=0)
# model.fit(X, y)
#
# # decision regions
# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X.values, y.values, model)
# plt.show()
#
# # drawing decision tree
# from dtreeplt import dtreeplt
# dtree = dtreeplt(model=model,feature_names=X.columns, target_names=["setosa","versicolor","virginica"])
# dtree.view()
# plt.show()
#
# print(pd.DataFrame(model.feature_importances_, X.columns))