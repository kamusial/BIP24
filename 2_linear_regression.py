import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('otodom.csv')
print(df.head(10).to_string())   #display all the columns
#print('iloc')
#print(df.iloc[:4, 1:4])  #df.iloc(rzędy, kolumny)

print(df.describe().to_string())
print('Now correlation')

print(df.iloc[:, 1:].corr())
sns.heatmap( df.iloc[:, 1:].corr(), annot=True )
plt.show()
sns.displot(df.price)
plt.show()
plt.scatter(df.space, df.price)
plt.show()
print(df.describe())

_min = df.describe().loc['min','price']
q1 = df.describe().loc['25%','price']
q3 = df.describe().loc['75%','price']
print(_min, q1, q3)

df1 = df[ (df.price >= _min) & (df.price <= q3) & (df.year < df.describe().loc['max','year'])]
#sns.displot(df1.price)
#plt.show()
print('New description')
print(df1.describe().to_string())
#traning data and test data
print('Now data')
print(df1.columns)
X = df1.iloc[:, 2: ]
y = df1.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.coef_)