import pandas as pd
import pytest
from pathlib import Path

from data.preprocess import create_features, create_target, process_csv, save_train_test_split

@pytest.fixture
def sample_data():
    data_source_filepath = "./tests/samples/test.csv"
    coolness_factor = 0.7
    return data_source_filepath, coolness_factor

def test_create_features():
    source_df = pd.DataFrame({'First Name': ['John', 'Jane'], 'Last Name Initial': ['D', 'S']})
    result_df = create_features(source_df)
    assert 'Length of First Name' in result_df.columns
    assert 'Distance' in result_df.columns


def test_create_target():
    features_df = pd.DataFrame({'Coolness': [50, 80]})
    coolness_factor = 70
    result_df = create_target(features_df, coolness_factor)
    assert 'is_Cool' in result_df.columns
    assert result_df['is_Cool'].tolist() == [0, 1]


def test_process_csv(sample_data):
    data_source_filepath, coolness_factor = sample_data
    result_df = process_csv(data_source_filepath, coolness_factor)
    assert 'Length of First Name' in result_df.columns
    assert 'Distance' in result_df.columns
    assert 'is_Cool' in result_df.columns


def test_save_train_test_split(tmpdir):
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    test_size = 0.2
    random_state = 42
    output_path = tmpdir.mkdir('output')
    train_data_path, test_data_path = save_train_test_split(df, test_size, random_state, output_path)
    assert Path(train_data_path).exists()
    assert Path(test_data_path).exists()
