import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def adj_r2(x,y):
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

data =pd.read_csv('Admission_Prediction.csv')
data.head()

data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())

data= data.drop(columns = ['Serial No.'])
data.head()

y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])

scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)

regression = LinearRegression()

regression.fit(x_train,y_train)


filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
a=loaded_model.predict(scaler.transform([[300,110,5,5,5,10,1]]))
a
regression.score(x_train,y_train)

adj_r2(x_train,y_train)
regression.score(x_test,y_test)
adj_r2(x_test,y_test)





