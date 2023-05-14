from flask import Flask, render_template, request
import csv
from io import StringIO
import pickle
import sys

from model_generator.preprocess import process_csv

# from model_generator.preprocess import process_csv


app = Flask(__name__)

# Load the trained model
model = None
with open('./src/model/random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input text from the form
        text1 = request.form['text1']
        text2 = request.form['text2']

        # Create an in-memory CSV file
        csv_data = StringIO()
        csv_writer = csv.writer(csv_data)
        csv_writer.writerow(['First Name', 'Last Name Initial'])
        csv_writer.writerow([text1, text2])
        csv_data.seek(0)

        # Process the in-memory CSV
        df = process_csv(csv_data)

        # Retrieve the processed values
        length_of_first_name = df.at[0, 'Length of First Name']
        distance = df.at[0, 'Distance']

        # Make a prediction using the preprocessed data
        score = model.predict([[length_of_first_name, distance]])

        return render_template('result.html', score=score[0])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)