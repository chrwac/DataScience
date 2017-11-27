## Some evaluations on the "Titanic data set",
## which is available on Kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

## read csv-file:
path = "C:\\Studium\\DataScience\\TitanicDataset\\"
path_training = path + "train.csv"
titanic_dataset = pd.read_csv(path_training,sep=",")

## some first information (number of samples, data-types, statistics ( mean,min,...))
print(titanic_dataset.describe())
print(titanic_dataset.dtypes)

## cleanup the data a little bit further:
## some of the "Age" - entries in the data set are missing, 
## replace the "NaNs" with the median-values 
## alternatives would be to drop the entire column "Age",
## or to drop all rows with missing values in the "Age"-column:
median_age = titanic_dataset["Age"].median()
titanic_dataset["Age"].fillna(median_age,inplace=True)

## (trying to get an intuition: Which features do matter ?)
## look at the survival probability overall:
num_samples = titanic_dataset["Survived"].count()
num_survived = np.sum(titanic_dataset["Survived"])
prob_survival = float(num_survived)/float(num_samples)
print("the overall survival probability is: ")
print(prob_survival) ## something on the order of 38.4 %

## does sex matter ? (i.e.: do men and women have significantly different survival probabilities ? )
num_women,num_men = titanic_dataset["Sex"].value_counts()["female"],titanic_dataset["Sex"].value_counts()["male"]
num_women_survived = np.sum(titanic_dataset["Survived"][titanic_dataset["Sex"]=="female"])
num_men_survived = np.sum(titanic_dataset["Survived"][titanic_dataset["Sex"]=="male"])
prob_survival_woman = float(num_women_survived)/float(num_women)
prob_survival_man  = float(num_men_survived)/float(num_men)
print("survivall probability of a woman:")
print(prob_survival_woman)	## something around 74.2 % --> so p(survival|woman) > p(survival)
print("survival probability of a man:")
print(prob_survival_man) ## 18.9 % --> p(survival|man) < p(survival)  , significant influence of gender on the survival probability

## similar analyses can be conducted for the other factors--> 
## age seems to play a role in the sense that the
## survival probability of very youn passengers is significantly above the average (>60% for passengers younger than 8)
## (kids were saved)
## AGE
age_means = np.zeros(10)
surv_probs = np.zeros(10)
for i in range(0,10):
    age_range = 8
    age_min = i*age_range
    age_max = (i+1)*age_range
    age_means[i] = (age_max+age_min)/2.0
    
    num_age_range = titanic_dataset["Survived"][(titanic_dataset["Age"]>age_min)&(titanic_dataset["Age"]<=age_max)].count()
    num_surv = np.sum(titanic_dataset["Survived"][(titanic_dataset["Age"]>age_min)&(titanic_dataset["Age"]<=age_max)])
    s_prob = float(num_surv)/float(num_age_range)
    surv_probs[i] = s_prob
    print("age from: ",age_min, "to: ", age_max)
    print("number of passengers in this category: ",num_age_range)
    print("survival probability: ",s_prob)
    print("")
## having SIBLINGS aboard helped ( p(survival|0 sibling)=34.5%)
## p(survival|1 sibling) =53.5 % (this goes down for more siblings, they probably were not able to watch out for everyone anymore --> but 2 people
## that watch out for each other was apparently ideal)

## having PARENTS or CHILDREN aboard had a similar effect

## travellin in a particular class helped strongly (first class > 60% survival probability, third class < 25 %)

## to build a decent classifier, one might thus try out to include:
## "Sex", "Class", "#Siblings aboard", "#Parents/Children aboard", probably "Age" and maybe also "Fare"- price

## factorize ("Sex"):
titanic_dataset["Sex"],sex_cat_enc = titanic_dataset["Sex"].factorize()
#rfc = RandomForestClassifier(n_estimators=300,max_depth=12,max_features=3,random_state=12)
rfc = RandomForestClassifier(n_estimators=1200,max_depth=9,max_features=3,random_state=10)
nnc = MLPClassifier(hidden_layer_sizes=(12,12),random_state=12,alpha=0.008)
lrc = LogisticRegression(C=0.4,random_state=12)
gnb = GaussianNB()

subset = ["Sex","Pclass","SibSp","Parch","Fare","Age"]
X = titanic_dataset[subset].as_matrix()
y = titanic_dataset["Survived"].as_matrix()
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.25,random_state=12)
#rfc.fit(X_train,y_train)
#y_train_preds = rfc.predict(X_train)
#acs_train = accuracy_score(y_train_preds,y_train)
#y_test_preds = rfc.predict(X_test)
#acs_test = accuracy_score(y_test_preds,y_test)

vcf = eclf2 = VotingClassifier(estimators=[
          ('rf', rfc)],
         voting='soft') #('lr', lrc),
vcf.fit(X_train,y_train.flatten())
preds_train_vcf = vcf.predict(X_train)
acs_train_vcf = accuracy_score(preds_train_vcf,y_train.flatten())
print(acs_train_vcf)
preds_test_vcf = vcf.predict(X_test)
print("accuracy on test set: ")
acs_test_vcf = accuracy_score(preds_test_vcf,y_test.flatten())
print(acs_test_vcf)

#print("accuracy score on training set: ")
#print(acs_train)
#print("accuracy score on test set:")
#print(acs_test)

## FINALLY: Load the actual test set and evaluate:
path_testing = path + "test.csv"
titanic_test_set = pd.read_csv(path_testing,sep=",")
titanic_test_set["Age"].fillna(median_age,inplace=True)
titanic_test_set["Sex"],sex_cat_test_enc = titanic_test_set["Sex"].factorize() 
median_fare = titanic_test_set["Fare"].median()
titanic_test_set["Fare"].fillna(median_fare,inplace=True)

X_test_test = titanic_test_set[subset].as_matrix()
y_test_test = vcf.predict(X_test_test)
d = {'PassengerId':titanic_test_set["PassengerId"],'Survived':y_test_test}
df = pd.DataFrame(d)
path_result = path + "result.csv"
df.to_csv(path_result,sep=",",index=False)

print(X)
