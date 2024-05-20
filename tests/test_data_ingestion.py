import pytest
from training_scripts.data_ingestion import get_train_test_data 
import pandas as pd


def test_get_train_test_data():
    data = pd.DataFrame({'col1': range(100), 'col2': [0] * 50 + [1] * 50,
                         'strength': range(100), 'password': range(100)})
    x_train, x_test, y_train, y_test = get_train_test_data(data, test_size=0.3,
                                                           target='col2')
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert x_train.shape == (70, 2)
    assert x_test.shape == (30, 2)
    assert y_train.shape == (70,)
    assert y_test.shape == (30,)
