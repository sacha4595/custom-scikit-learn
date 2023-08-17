import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted




def get_feature_importance_minimal_depth(forest: RandomForestClassifier) -> np.ndarray:

    """
    Inputs : A random forest (list of decision trees)
    Outputs : (n_features) array of average minimal depths for each feature over the forest
    """

    check_is_fitted(forest, 'estimators_')

    if type(forest.estimators_) is not list:
        forest_estimators = forest.estimators_.flatten().tolist()
    else:
        forest_estimators = forest.estimators_

    n_trees = forest.n_estimators
    n_features = forest_estimators[0].tree_.n_features
    if n_features == 1:
        return np.array([0])
    minimal_depths = np.zeros((n_trees, n_features))

    for i, tree in enumerate(forest_estimators):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        features = tree.tree_.feature

        feature_minimal_depth_tree = np.array(n_features*[tree.tree_.max_depth])

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:

            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            if depth < feature_minimal_depth_tree[features[node_id]]:
                feature_minimal_depth_tree[features[node_id]] = depth

            is_split_node = children_left[node_id] != children_right[node_id]

            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        minimal_depths[i,:] = feature_minimal_depth_tree

    minimal_depths_average = np.mean(minimal_depths, axis=0)

    minimal_depths_average = minimal_depths_average / np.max(minimal_depths_average)

    minimal_depths_average = 1 - minimal_depths_average

    return minimal_depths_average

