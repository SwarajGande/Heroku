import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("hiring.csv")
df["experience"].fillna(0,inplace = True)
df["test_score"].fillna(df["test_score"].mean(),inplace = True)
def string_to_number(word):
    dict = {0:0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"zero":0,"twelve":12}
    
    return dict[word]
df["experience"] = df["experience"].apply(lambda x:string_to_number(x))

x = df.drop("salary",axis = 1)
x = np.array(x).reshape(-1,3)
y = df["salary"]
y = np.array(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.10,random_state = 1)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)
print(model.predict([[5,8,7]]))

import pickle
pickle.dump(model,open("my_model.pkl","wb"))