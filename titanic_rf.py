# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
import pydotplus

dataset = pd.read_csv('C:\Users\Vignesh\Downloads\\test.csv')
orig_dataset = dataset
dataset = dataset.drop('PassengerId',1) #Taking out column PassengerID
dataset = dataset.drop('Name',1)
dataset = dataset.drop('Ticket',1)
dataset = dataset.drop('Cabin',1)
print dataset
dataset.Age = dataset.Age.fillna(dataset.mean()['Age'])
dataset.Fare = dataset.Fare.fillna(dataset.mean()['Fare']) #filling missing values with mean
print dataset
#print dataset.Embarked.mode()
dataset.Embarked = dataset.Embarked.fillna(dataset.Embarked.mode())
pd.set_option('display.max_rows', None)
print dataset
dataset.Age = dataset.Age.astype(int) #converting float values into int
#print dataset.Cabin.str[0] #getting the first chracter within 'Cabin' column
print dataset
new_column = (pd.get_dummies(dataset.Sex, drop_first = True).astype(int))
#print new_column
dataset = dataset.join(new_column)
dataset = dataset.drop('Sex',1)
embarked_columns= pd.get_dummies(dataset.Embarked, drop_first = False, prefix = 'Embark').astype(int)
#print embarked_columns
dataset = dataset.join(embarked_columns)
dataset = dataset.drop('Embarked',1)
print dataset
"""rowcount = len(dataset.index)
print rowcount
Y = dataset.Survived
Y_train = Y.head((rowcount/2))
Y_test = Y.tail((rowcount/2)+1)
print Y_train
print Y_test
X = dataset
X = X.drop('Survived',1)
X_train = X.head((rowcount/2))
X_test = X.tail((rowcount/2)+1)
print X_test
print dataset
#print X
#print Y"""
clf = ensemble.RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=9, min_samples_split=1, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
clf = clf.fit(X_train,Y_train)
pred_Y_train = clf.predict(X_train)
print dataset
pred_Y_test = clf.predict(dataset)
#print pred_Y_test
Survived = pd.DataFrame({'Survived':pred_Y_test})
print Survived
'''orig_dataset = orig_dataset.join(Survived)
print orig_dataset'''
Survived.to_csv(path_or_buf = 'C:\Users\Vignesh\Downloads\\test_pred_rf.csv')
#print "Confusion Matrix:\n", metrics.confusion_matrix(Y,pred_Y,labels=None)
print "Training Confusion Matrix:"
pd.crosstab(Y_train, pred_Y_train, rownames=['True'], colnames=['Predicted'], margins=True)
print "Training Accuracy: ",metrics.accuracy_score(Y_train,pred_Y_train)*100,"%"
print "Testing Confusion Matrix:"
pd.crosstab(Y_test, pred_Y_test, rownames=['True'], colnames=['Predicted'], margins=True)
print "Testing Accuracy: ",metrics.accuracy_score(Y_test,pred_Y_test)*100,"%"
dot_data = tree.export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("C:\\tree.pdf")
