import csv
import random
import string
import names

def generate_first_name():
    '''Generate random first name'''
    return ''.join(names.get_first_name())

# 
def generate_last_name_initial():
    '''Generate random letter for last name initial'''
    return random.choice(string.ascii_uppercase)

def generate_coolness():
    '''Generate random coolness value (integer from 0 to 100)'''
    return random.randint(0, 100)

def generate_csv(num_rows, output_filepath):
    '''Generate CSV file with first name, last initial, and a coolness score'''

    # Create the CSV file
    with open(output_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['First Name', 'Last Name Initial', 'Coolness'])
        
        for _ in range(num_rows):
            first_name = generate_first_name()
            last_name_initial = generate_last_name_initial()
            coolness = generate_coolness()
            
            writer.writerow([first_name, last_name_initial, coolness])

    print(f'CSV file created successfully at {output_filepath} with {num_rows} rows.')

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)
    num_rows = config['data_generator']['rows']
    output_filepath = config['data_generator']['output_filepath']

    generate_csv(num_rows, output_filepath)
