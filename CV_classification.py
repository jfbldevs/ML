from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
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
