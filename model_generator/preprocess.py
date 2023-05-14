import pandas as pd

def process_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Calculate the length of the first name and add a column
    df['Length of First Name'] = df['First Name'].apply(lambda x: len(x))

    # Calculate the distance between the first letters of the first and second columns and add a column
    df['Distance'] = df.apply(lambda row: abs(ord(row['First Name'][0]) - ord(row['Last Name Initial'])), axis=1)

    return df

if __name__ == '__main__':
    import sys

    # Check if the CSV file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print('Please provide the filepath of the CSV file.')
        sys.exit(1)

    filepath = sys.argv[1]
    dataframe = process_csv(filepath)
    print(dataframe)