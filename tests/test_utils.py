from training_scripts.utils.find_class_weights import find_class_weights
from training_scripts.utils.logger import Logger
import pandas as pd


def test_logger():
    logger = Logger()
    res = logger.log_message('init')
    assert res is None


def test_class_weights():
    y = pd.Series([0, 0, 1, 1, 1, 0])
    class_weights = find_class_weights(y)
    assert class_weights == {0: 1.0, 1: 1.0}
