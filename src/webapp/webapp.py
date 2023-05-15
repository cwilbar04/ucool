import pandas as pd
import os
from flask import Flask, render_template, request, redirect

from model.score import score_input
from helpers import read_params

app = Flask(__name__)

app.config['CONFIG_FILE'] = os.environ.get('CONFIG_FILE', 'params.yml')
app.config['RUN_TYPE'] = os.environ.get('RUN_TYPE', 'DEV')
config = read_params(app.config['CONFIG_FILE'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input text from the form
        first_name = request.form['text1']
        last_initial = request.form['text2']

        # Create a dataframe from the text input
        input_df=pd.DataFrame({'First Name':[first_name], 'Last Name Initial':[last_initial]})

        _, score_meaning = score_input(config, input_df, app.config['RUN_TYPE'])

        return render_template('result.html', score_meaning=score_meaning)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)