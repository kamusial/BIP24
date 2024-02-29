import pandas as pd    #'as'  alias
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
print(type(df))
print(df)
print(df.head(5))
print(df.Gender.value_counts())
df.Height *= 2.54
df.Weight /= 2.2
print(df.head(5))
sns.displot(df.Weight)  # men and women together
# sns.displot(df.query("Gender=='Male'").Weight)
# sns.displot(df.query("Gender=='Female'").Weight)
plt.show()

#gender - not numeric data
df = pd.get_dummies(df)
del(df["Gender_Male"])
df.rename(columns={'Gender_Female': 'Gender'}, inplace=True)
print(df.head(5))

model = LinearRegression()
model.fit(df[ ['Height', 'Gender'] ], df['Weight'] )
print(model.coef_, model.intercept_)
print('Eq: Height * ',model.coef_[0], '+ Gender * ',model.coef_[1],' = Weight')

# self formula
gender = 0
height = 170
weight = model.intercept_ + model.coef_[0] * height + model.coef_[1] * gender
print(weight)