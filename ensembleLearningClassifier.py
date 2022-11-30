# example of evaluating a bagging ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier


class ensembleClassifier:
    def __init__(self, X,y):
        self.X=X
        self.y=y

    def bagging(self):
        model = BaggingClassifier(n_estimators=50)
        # configure the resampling method
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the ensemble on the dataset using the resampling method
        n_scores = cross_val_score(model, self.X, self.y, scoring='accuracy', cv=cv, n_jobs=-1)
