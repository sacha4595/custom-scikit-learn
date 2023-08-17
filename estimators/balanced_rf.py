# This script implements a custom bootstrap method for random forest classification.

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.class_weight import compute_sample_weight



class CustomBootstrapRandomForestClassifier(RandomForestClassifier,BaseEstimator):

    def __init__(self, estimator = RandomForestClassifier(), bootstrap_ratio = 1):

        self.bootstrap_ratio = bootstrap_ratio
        self.estimator = estimator
        self.bootstrap_method = bootstrap_ratio
        for key, value in estimator.get_params().items():
            setattr(self, key, value)

    def _bootstrap_sample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create a bootstrap sample of the data with under sampling of the majority class and over sampling with replacement of the minority class.

        """
        # Find majority class, 0 or 1
        counts = np.bincount(y.astype(int))
        minority_class = np.argmin(counts)
        n_minority = counts[minority_class]
        n_majority = counts[1 - minority_class]

        n_size = int(n_minority * self.bootstrap_ratio)

        # bootstrap minority class
        indices_minority = np.random.choice(
            np.where(y == minority_class)[0],
            size = n_size,
            replace = True)

        # bootstrap majority class
        indices_majority = np.random.choice(
            np.where(y != minority_class)[0],
            size = n_size,
            replace = True)


        indices = np.hstack([indices_majority, indices_minority])
        np.random.shuffle(indices) # in-place
        
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y[indices]

        return X_bootstrap, y_bootstrap

    def fit(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=True)

        self.bootstrap = True

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        assert isinstance(y, np.ndarray)

        if self.random_state is not None:
            np.random.seed(self.estimator.random_state)

        trees = list()
        for i in range(self.estimator.n_estimators):

            # Get our bootstrapped data
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
        
            # Fit a decision tree
            tree = DecisionTreeClassifier(
              max_depth = self.max_depth,
              min_samples_leaf = self.min_samples_leaf,
              max_features = self.max_features,
              random_state = self.random_state+i
            )
            tree.fit(X_bootstrap, y_bootstrap)
            trees.append(tree)

        self.estimators_ = trees

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.is_fitted_ = True



        return self
