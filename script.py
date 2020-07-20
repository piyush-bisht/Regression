import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df=pd.read_csv('tennis_stats.csv')
print(df.head())

model=LinearRegression()
#model.fit(df[['FirstServe']],df[['Wins']])


plt.scatter(df[['BreakPointsOpportunities']],df[['Wins']])
plt.show()

y=df[['Wins']]
x=df[['BreakPointsOpportunities']]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))
y_predict=model.predict(x_test)
plt.scatter(x_test,y_predict,alpha=0.4)
plt.xlabel("BreakPointsOpportunities")
plt.ylabel("Wins")
plt.show()






# perform exploratory analysis here:
model2=LinearRegression()
x=df[['TotalPointsWon']]
y=df[['Wins']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8 , test_size=0.2)
model2.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("model2 score",model2.score(x_test,y_test))
plt.scatter(x_test,y_predict)
plt.xlabel("FirstServe")
plt.ylabel("Wins")
plt.show()




















## perform single feature linear regressions here:






















## perform two feature linear regressions here:

model4=LinearRegression()
x=df[['FirstServeReturnPointsWon','FirstServePointsWon','ReturnGamesWon','ServiceGamesWon','BreakPointsOpportunities']]
y=df[['Winnings']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
model4.fit(x_train,y_train)
predicted=model4.predict(x_test)
print("model4 score",model4.score(x_test,y_test))


model5=LinearRegression()
x=df[['FirstServeReturnPointsWon','BreakPointsOpportunities']]
y=df[['Winnings']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
model5.fit(x_train,y_train)
predicted=model5.predict(x_test)
print("model5 score",model5.score(x_test,y_test))




















## perform multiple feature linear regressions here:





















