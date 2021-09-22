from sklearn import datasets
import pandas as pd
#iris = datasets.load_iris()
#X, Y = iris.data[:, [2,3]], iris.target
datos=pd.read_csv("/content/classtest.csv")

#print("Dataset Features : ", iris.feature_names)
#print("Dataset Target : ", iris.target_names)
#print('Dataset Size : ', X.shape, Y.shape)
X=datos[['KIDA850101','FAUJ830101','QIAN880128','MIYS990105','MIYS990104','RADA880108','COWR900101','ROSG850102','VINM940101','GUYH850102']]
y = datos['PRED']

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# Evaluate model: 10-fold cross-validation
scores1 = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
scores2 = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
scores3 = cross_val_score(model, X, y, scoring='precision', cv=cv)
scores4 = cross_val_score(model, X, y, scoring='recall', cv=cv)
scores5 = cross_val_score(model, X, y, scoring='f1', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('ROC: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Precision: %.3f (%.3f)' % (mean(scores3), std(scores3)))
print('Recall: %.3f (%.3f)' % (mean(scores4), std(scores4)))
print('F1: %.3f (%.3f)' % (mean(scores5), std(scores5)))

y
