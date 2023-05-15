import pandas as pd
import os
from flask import Flask, render_template, request
import mlflow.pyfunc

from data.preprocess import create_features
from helpers import read_params

app = Flask(__name__)

app.config['CONFIG_FILE'] = os.environ.get('CONFIG_FILE', 'params.yml')
config = read_params(app.config['CONFIG_FILE'])
mlflow_config = config['model']['mlflow_config']

# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])
# Load the trained model
model = mlflow.pyfunc.load_model("models:/is_you_cool_classifier/Staging")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input text from the form
        first_name = request.form['text1']
        last_initial = request.form['text2']

        # Create a dataframe from the text input
        input_df=pd.DataFrame({'First Name':[first_name], 'Last Name Initial':[last_initial]})

        # Add the features to the dataframe
        features_df = create_features(input_df)

        # Retrieve the processed values
        # length_of_first_name = df.at[0, 'Length of First Name']
        # distance = df.at[0, 'Distance']

        # Make a prediction using the preprocessed data
        score = model.predict(features_df)

        return render_template('result.html', score=score[0])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)