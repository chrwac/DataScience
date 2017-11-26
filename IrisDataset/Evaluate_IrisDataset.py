## Uses GridSearchCV on stratified subsets to
## find optimal values for a RandomForestClassifier on
## a two-dimensional subset (sepal width, petal length)
## of the Iris - data set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

data = datasets.load_iris()
print(data.feature_names)
print(data.target_names)

## train on the features 2 and 3 (indices 1 and 2) only:
## (only to better visualize the result in 2D)
X_train = data.data[:,1:3]
y_train = data.target
rfc = RandomForestClassifier()
param_grid = [{"n_estimators":[10,20,50],"max_depth":[4,8,16]}]
grid_search = GridSearchCV(rfc,param_grid,cv=5,scoring="accuracy")
grid_search.fit(X_train,y_train)

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"],cv_res["params"]):
    print(mean_score,params)

## create a meshgrid to plot the decision boundaries of the random forest classifier
x_min,x_max = min(X_train[:,0]),max(X_train[:,0])
y_min,y_max = min(X_train[:,1]),max(X_train[:,1])
dx = 0.1*(x_max-x_min)
dy = 0.1*(y_max-y_min)
xs = np.arange(x_min-dx,x_max+dx,dx/10)
ys = np.arange(y_min-dy,y_max+dy,dy/10)

num_pts = 100
xx,yy = np.meshgrid(xs,ys)
Z = grid_search.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig=plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
plt.contourf(xx, yy, Z, alpha=0.99)

symbols = ("s","D","o")
for i in range(0,len(y_train)):
	plt.scatter(X_train[i,0],X_train[i,1],marker=symbols[int(y_train[i])],c="yellow",s=40)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal length (cm)")
## plot the actual data points, 
fig.savefig('C:/Studium/DataScience/IrisDataset/test.png')   # save the figure to file
plt.close(fig)  

