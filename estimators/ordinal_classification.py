# This class implements a method for ordinal regression compatible with the scikit-learn API.
# It is based on the paper:
# A Simple Approach to Ordinal Classification
# Eibe Frank, Mark Hall
# DOI:10.1007/3-540-44795-4_13

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


from importance_getter import get_feature_importance_minimal_depth


class OrdinalClassification(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, ordered_classes):
        '''estimator: a scikit-learn estimator
        ordered_classes: a list of ordered classes on which the ordinal classification is based
        '''
        self.estimator = estimator
        self.ordered_classes = ordered_classes
        self.selected_features = None

    def fit(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=True)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        assert isinstance(y, np.ndarray)

        if self.estimator.random_state is not None:
            np.random.seed(self.estimator.random_state)


        n_sub_learner = len(self.ordered_classes) - 1
        targets = np.zeros((n_sub_learner, len(y)))
        self.estimators = np.ndarray(n_sub_learner, dtype=object)
        feature_importances_ = np.zeros((n_sub_learner, X.shape[1]))
        feature_importances_minimal_depth_ = np.zeros((n_sub_learner, X.shape[1]))

        for i, target_value in enumerate(self.ordered_classes[:-1]):
            targets[i,:] = [0 if self.ordered_classes.index(x) <= self.ordered_classes.index(target_value) else 1 for x in y]
            sub_learner = clone(self.estimator)
            sub_learner.fit(X, targets[i,:])
            self.estimators[i] = sub_learner

            feature_importances_[i,:] = sub_learner.feature_importances_
            if hasattr(sub_learner, 'estimators_'):
                feature_importances_minimal_depth_[i,:] = get_feature_importance_minimal_depth(sub_learner)

        self.feature_importances_ = np.max(feature_importances_, axis=0)
        if hasattr(sub_learner, 'estimators_'):
            self.feature_importances_minimal_depth_ = np.min(feature_importances_minimal_depth_, axis=0)



        self.is_fitted_ = True

        return self
    
    
    def predict_proba(self, X):

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        n_classes = len(self.ordered_classes)
        n_sub_learner = n_classes - 1
        n_samples = X.shape[0]


        predictions = np.zeros((n_samples, n_classes))
        for i in range(n_sub_learner+1):
            if i == 0:
                predictions[:,i] = 1-self.estimators[i].predict_proba(X)[:,list(self.estimators[i].classes_).index(1)]
            if i == n_sub_learner:
                predictions[:,i] = self.estimators[i-1].predict_proba(X)[:,list(self.estimators[i-1].classes_).index(1)]
            if i > 0 and i < n_sub_learner:
                predictions[:,i] = np.abs(self.estimators[i-1].predict_proba(X)[:,list(self.estimators[i-1].classes_).index(1)] - self.estimators[i].predict_proba(X)[:,list(self.estimators[i].classes_).index(1)])
        return predictions
    
    def predict_risk(self, X):

        check_is_fitted(self, 'is_fitted_')

        pred = self.predict_proba(X)

        n_samples = X.shape[0]

        def predicted_risk(x):
            n = len(x)
            score = 0
            for i in range(n):
                score += x[i] * (i/n)

            return score
            
        risk = np.zeros(n_samples)
        # final_prediction_class = np.zeros(n_samples)

        for i in range(n_samples):
            risk[i] = predicted_risk(pred[i,:])
        
        return risk
    
    def predict(self, X):
        return self.predict_risk(X)
    
    def score(self, X, y):
        '''
        Compute Accuracy Score
        '''

        X, y = check_X_y(X, y, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        pred = self.predict_risk(X)

        y = [1 if self.ordered_classes.index(x) == len(self.ordered_classes)-1 else 0 for x in y]
        return roc_auc_score(y, pred)