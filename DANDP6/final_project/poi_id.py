#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from time import time
from scipy.stats import uniform as sp_rand
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary'] # You will need to use more features

all_features = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 
'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Credit to https://github.com/willemolding/IdentifyFraudEnron/blob/master/poi_id.py
#Using pandas to create a dataframe.
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

#Credit to my grader for the code below to provide some stats into the dataset.
print "Number POI: ", sum(df['poi'])
print "Number non-POI: ", len(df['poi'])-sum(df['poi'])

#Credit to https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py
#The following statements are to view the data. More specifically, look at the length
#and all values in the dictionary.
print "Total values: ", len(data_dict.keys())
#print data_dict.values()

#Provided by my grader to get features with missing values.
def value_missing(x):
	return sum(x.isnull())
	column_nan = df.apply(val_missing, axis = 0)
	column_nan.sort_values(ascending = False)
	return value_missing

### Task 2: Remove outliers
#Credit to https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py
#The library features contains "salary" and "bonus". Then, take out the first value of
#'TOTAL'. Finally, use featureFormat() to format the dictionary, by features. By default,
#featureFormat() will remove all zeroes.
feature = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, feature)

#Select best features
#array = df.values
#print array
#X = array[:,:]
#Y = array[:,:]
#test = SelectKBest(score_func = chi2, k = 4)
#fit = test.fit(X,Y)
#numpy.set_printoptions(precision=3)
#print(fit.scores_)
#features = fit.transform(X)
#print(features[0:2,:])


#Credit to https://github.com/willemolding/IdentifyFraudEnron/blob/master/poi_id.py
#Create dataframes and remove any NA's.
df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())


#Credit to https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py
#Creating a list called outliers. For each key in the dictionary, loop through
#the values, and look at the salary. If the salary is NaN, then continue to add
#the key with the integer of the value.
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))


#Credit to https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py
#Create a plot for features. Loop through data and create a scatterplot, showing
#salary on the x-axis, and bonus on the y-axis.
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

### Task 3: Create new feature(s)

#Credit to https://github.com/willemolding/IdentifyFraudEnron/blob/master/poi_id.py
df['fraction_to_poi_email'] = df.from_this_person_to_poi / df.from_messages
df['fraction_from_poi_email'] = df.from_poi_to_this_person / df.to_messages

#features_list = [u'poi',u'salary', u'bonus', u'expenses', 
#u'from_poi_to_this_person', u'from_this_person_to_poi', u'shared_receipt_with_poi',
#u'from_messages', u'to_messages', u'fraction_to_poi_email', u'fraction_from_poi_email']

features_list = [u'exercised_stock_options', u'total_stock_value', u'long_term_incentive', u'from_messages', u'deferral_payments', u'poi', u'total_payments', u'director_fees', u'salary']

print "Fraction to POI email: ", sum(df['fraction_to_poi_email'])
print "Fraction from POI email: ", sum(df['fraction_from_poi_email'])

### Store to dataset for easy export below.
dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(dataset, features_list, sort_keys = True)
#my_data = featureFormat(dataset, features_list)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#Credit to https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py
#First, use cross validation and train_test_split() to take in features and labels
#with a proportion of 0.1 of the test size, with 42 as the random state. We assign
#those to the various training and testing variables.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, 
	labels, test_size=0.1, random_state=42)

#Then, we use KFold to split the data into 3 consecutive folds.
# We then assign to kf to create out training and testing sets.
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]


#Finally, we use DecisionTreeClassifier to create a model that predicts the value
#of a target variable by learning from the data features.
t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning (decision tree): ', score
scores = clf.score(features_train,labels_train)
print 'accuracy after tuning (decision tree): ', scores
print "GaussianNB algorithm time:", round(time()-t0, 3), "s"

#Use GaussianNB
claf = GaussianNB()
claf.fit(features_train,labels_train)
scor = claf.score(features_test,labels_test)
print 'accuracy before tuning (GaussianNB): ', scor
scors = claf.score(features_train,labels_train)
print 'accuracy after tuning (GaussianNB): ', scors
print "GaussianNB algorithm time:", round(time()-t0, 3), "s"

#Support vector machines
clasf = svm.SVC()
clasf.fit(features_train,labels_train)
scre = clasf.score(features_test,labels_test)
print 'accuracy before tuning (SVM): ', scre
scres = clasf.score(features_train,labels_train)
print 'accuracy after tuning (SVM): ', scres
print "SVM algorithm time:", round(time()-t0, 3), "s"

#Feature selection by suggest of grader. Credit to http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, random_state=0, shuffle=False)
#cloud = ExtraTreesClassifier(n_estimators=250, random_state=0)
#cloud.fit(X,y)
#importances = clf.feature_importances_
#indices = np.argsort(importances)[::-1]
#print "Feature ranking: "
#for f in range(X.shape[1]):
#   print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, dataset, features_list)
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )