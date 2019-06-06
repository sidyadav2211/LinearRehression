import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("hours.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

dataset.head()

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)


#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
regressor.coef_
regressor.intercept_
print ("Acurracy: ",regressor.score(X,y) * 100)

y_pred=regressor.predict([[8]])
print(y_pred)

hours=int(input("Enter number of hours: "))
eq=regressor.coef_*hours+regressor.intercept_
y_pred = regressor.predict([[0]])
print(y_pred)

print('y = %f*%f+%f' %(regressor.coef_,hours,regressor.intercept_))
print('Risk Score: ',eq[0])

plt.plot(X,y,'o')
plt.plot(X, regressor.predict(X));
plt.show()
