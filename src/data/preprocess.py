import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def create_features(df):
    '''
    Function to create features based on input dataframe with columns 'First Name' & 'Last Name Initial'

    Returns dataframe obejct with features included.

    IMPORTANT: Make sure to add feature to the data.target in params.yml if desired to be used in the model.
    '''
     # Calculate the length of the first name and add a column
    df['Length of First Name'] = df['First Name'].apply(lambda x: len(x))

    # Calculate the distance between the first letters of the first and second columns and add a column
    df['Distance'] = df.apply(lambda row: abs(ord(row['First Name'][0]) - ord(row['Last Name Initial'])), axis=1)

    return df

def create_target(df, coolness_factor):
    '''
    Inputs dataframe with 'Coolness' column.

    Returns dataframe with specified target to model.
    '''
    df['is_Cool'] = df['Coolness'].apply(lambda x: 1 if x > coolness_factor else 0)
    return df

def process_csv(data_source_filepath, coolness_factor):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_source_filepath)

    # Add features
    df = create_features(df)

    # Add Classifier Categories
    df = create_target(df, coolness_factor)

    return df

def save_train_test_split(df, test_size, random_state, output_path):

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

    # Save formatted_datetime back to params 

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)
    coolnes_factor = config['data']['coolness_factor']
    
    # Use demo filepath for testing from command line
    data_source_filepath = config['data_generator']['output_filepath']    

    dataframe = process_csv(data_source_filepath, coolnes_factor)
    print(dataframe)