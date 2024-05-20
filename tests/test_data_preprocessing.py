from training_scripts.data_preprocessing import DataPreprocessor
import pandas as pd


def test_data_preprocessing():
    data = pd.DataFrame({'password': ['Abc123D!@', '12A'],
                         'strength': [2, 0]})
    data_preprocessor = DataPreprocessor()
    _ = data_preprocessor.fit(data)
    transformed_data = data_preprocessor.transform(data)
    expected_output = pd.DataFrame({'password': ['Abc123D!@', '12A'],
                                    'strength': [2, 0],
                                    'len': [9, 3],
                                    'num_count': [3, 2],
                                    'uppercase_count': [2, 1],
                                    'special_char_count': [2, 0]})
    assert transformed_data.equals(expected_output)
