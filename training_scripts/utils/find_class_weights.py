from sklearn.utils import class_weight
import pandas as pd


def find_class_weights(y: pd.Series):
    """
    - Function to return class weights to be used for cost-sensitive learning.
    - Returns a dictionary of classes as keys and weights as values.
    - Scikit-Learn compute_class_weight returns an array of class weights which is not compatible
      with ML algorithms.
    """
    classes_ = sorted([*y.unique()]).copy()  # returns sorted unique class labels
    # Dictionary comprehension that assigns each class (key) its weight (value)
    class_weight_param = {class_: weight for class_, weight in
                          zip(classes_, class_weight.compute_class_weight(class_weight='balanced',
                                                                          classes=classes_, y=y))}
    return class_weight_param
