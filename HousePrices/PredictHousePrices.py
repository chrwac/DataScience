import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
## Calculate the root-mean-squared log error:
def CalculateRMSLE(y_preds,y_real):
	length = len(y_preds)
	res = 0
	for i in range(0,length):
		res = res + (np.log(y_preds[i]+1)-np.log(y_real[i]+1))**2
	res = np.sqrt(res/float(length))
	return(res)



class EvaluateHousingPrices:
	def __init__(self,path):
		self.__path = path
	def LoadTrainingDataFromCSV(self,filename,sep=","):
		comp_path_train_data = self.__path + filename
		self.__pdf_train = pd.read_csv(comp_path_train_data,sep)
	def TrainigDataPrintInfo(self):
		print(self.__pdf_train.describe())
		print(self.__pdf_train.dtypes)
	def SetTargetAndFeatureNames(self,target_name,feature_names):
		self.__target_name = target_name
		self.__feature_names = feature_names
	def PrepareTrainData(self):
		self.__yvalues_train_dataset = self.__pdf_train[self.__target_name].as_matrix().reshape(-1,1)
		self.__xvalues_train_dataset = self.__pdf_train[self.__feature_names]
		
		## include object data types:
		object_datatypes = self.__xvalues_train_dataset.select_dtypes(include=['object']).columns
		for i in object_datatypes:
			self.__xvalues_train_dataset[i],enc = self.__xvalues_train_dataset[i].factorize()
		print(self.__xvalues_train_dataset)

		## now convert to matrix and apply imputer:
		self.__xvalues_train_dataset = self.__xvalues_train_dataset.as_matrix()
		self.__imp = Imputer(strategy="median")
		self.__xvalues_train_dataset = self.__imp.fit_transform(self.__xvalues_train_dataset)
		self.__stdscaler = StandardScaler()
		self.__xvalues_train_dataset = self.__stdscaler.fit_transform(self.__xvalues_train_dataset)
		print(self.__xvalues_train_dataset)
		## scale x-values:
	def FitAndEvaluateTrainingData(self):
		self.__reg = GradientBoostingRegressor(loss="ls")#linear_model.Ridge (alpha = 0.1)#LinearRegression()
		param_grid = [{"learning_rate":[0.03,0.1,0.2],"n_estimators":[50,100,150]}]
		grid_search = GridSearchCV(estimator=self.__reg,param_grid=param_grid,cv=5,scoring="neg_mean_squared_log_error")
		grid_search.fit(self.__xvalues_train_dataset,self.__yvalues_train_dataset.ravel())
		cvres = grid_search.cv_results_

		for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
			print(np.sqrt(-mean_score),params)
		print("best parameters:")
		print(grid_search.best_params_)
		self.__reg = grid_search.best_estimator_ 
	def LoadTestDataFromCSV(self,filename,sep=","):
		comp_path_test_data = self.__path + filename
		self.__pdf_test = pd.read_csv(comp_path_test_data,sep)
	def PrepareTestData(self):
		self.__xvalues_test_dataset = self.__pdf_test[self.__feature_names]
		object_datatypes = self.__xvalues_test_dataset.select_dtypes(include=['object']).columns
		for i in object_datatypes:
			self.__xvalues_test_dataset[i],enc = self.__xvalues_test_dataset[i].factorize()
		self.__xvalues_test_dataset = self.__imp.transform(self.__xvalues_test_dataset)
		self.__xvalues_test_dataset = self.__stdscaler.transform(self.__xvalues_test_dataset)
		#print(self.__xvalues_test_dataset)
	def PredictTestSet(self):
		self.__preds_y = self.__reg.predict(self.__xvalues_test_dataset)
		return self.__preds_y
	def WriteTestSet(self,filename):
		## CHANGE THIS EVENTuALLY:
		self.__preds_y[self.__preds_y<0]=0
		d = {'Id':self.__pdf_test["Id"],'SalePrice':self.__preds_y.flatten()}
		df = pd.DataFrame(d)
		path_result = self.__path + filename
		df.to_csv(path_result,sep=",",index=False)




path = "C:\\Studium\\DataScience\\HousePrices\\"
ehp = EvaluateHousingPrices(path)
ehp.LoadTrainingDataFromCSV("train.csv")
ehp.TrainigDataPrintInfo()
#feature_names = ["YrSold","LotArea","BedroomAbvGr","KitchenAbvGr","OverallQual","OverallCond","KitchenQual","PoolArea","PoolQC","GrLivArea","GarageArea","GarageQual","GarageCond","YearBuilt","OpenPorchSF","WoodDeckSF"]
feature_names = ["OverallQual","GarageCars","GrLivArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd","Fireplaces","BsmtFinSF1","WoodDeckSF","2ndFlrSF","OpenPorchSF","HalfBath","LotArea","BsmtFullBath"]
ehp.SetTargetAndFeatureNames("SalePrice",feature_names)
ehp.PrepareTrainData()
ehp.FitAndEvaluateTrainingData()
ehp.LoadTestDataFromCSV("test.csv")
ehp.PrepareTestData()
preds_y = ehp.PredictTestSet()
ehp.WriteTestSet("result2.csv")


#print(preds_y[preds_y<0])
