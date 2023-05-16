from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def create_features(source_df):
    '''
    Function to create features based on input dataframe with columns 'First Name' & 'Last Name Initial'

    Args:
        source_df (DataFrame): Input DataFrame with columns 'First Name' and 'Last Name Initial'.

    Returns:
        DataFrame: DataFrame object with features included.

    IMPORTANT: Make sure to add feature to the model.features in params.yml if desired to be used in the model.
    '''
    # Calculate the length of the first name and add a column
    source_df['Length of First Name'] = source_df['First Name'].apply(len)

    # Calculate the distance between the first letters of the first and second columns and add a column
    source_df['Distance'] = source_df.apply(lambda row: abs(ord(row['First Name'][0]) - ord(row['Last Name Initial'])), axis=1)

    # Calculate the distance between the first letters of the first name and last letter of first name
    source_df['Distance'] = source_df.apply(lambda row: abs(ord(row['First Name'][0]) - ord(row['First Name'][-2])), axis=1)

    return source_df

def create_target(features_df, coolness_factor):
    '''
    Create target variable based on 'Coolness' column.

    Args:
        features_df (DataFrame): DataFrame with the 'Coolness' column.
        coolness_factor (float): Threshold value to determine if a value is considered 'Cool'.

    Returns:
        DataFrame: DataFrame with the specified target variable.

    IMPORTANT: Make sure to add target to the model.target in params.yml if desired to be used in the model.
    '''
    features_df['is_Cool'] = features_df['Coolness'].apply(lambda x: 1 if x > coolness_factor else 0)
    return features_df

def process_csv(data_source_filepath, coolness_factor):
    '''
    Read CSV data, create features, and generate the target variable.

    Args:
        data_source_filepath (str): Filepath to the CSV data source.
        coolness_factor (float): Threshold value to determine if a value is considered 'Cool'.

    Returns:
        DataFrame: Processed DataFrame with features and target variable.

    '''
    # Read the CSV file into a pandas DataFrame
    source_df = pd.read_csv(data_source_filepath)

    # Add features
    features_df = create_features(source_df)

    # Add Classifier Categories
    full_df = create_target(features_df, coolness_factor)

    return full_df

def save_train_test_split(df, test_size, random_state, output_path):
    '''
    Split data into train and test sets, save them as CSV files, and return the filepaths.

    Args:
        df (DataFrame): Input DataFrame to be split into train and test sets.
        test_size (float): Proportion of the dataset to include in the test set.
        random_state (int): Seed for the random number generator.
        output_path (str): Directory path to save the train and test data.

    Returns:
        str: Filepath of the saved train data.
        str: Filepath of the saved test data.
    '''

    # Split the data into train and test sets
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Get current datetime to append to output path
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.replace(microsecond=0).isoformat()

    # Generate train and test data path
    train_data_path = f'{output_path}/train_{formatted_datetime}.csv'
    test_data_path = f'{output_path}/test_{formatted_datetime}.csv'

    # Save data to output path for modeling usage
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

    return train_data_path, test_data_path

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)

    dataframe = process_csv(config['data_generator']['output_filepath'], config['data']['coolness_factor'])
    print(dataframe)