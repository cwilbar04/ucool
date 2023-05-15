import mlflow
import pandas as pd

from data.preprocess import create_features

def score_input(config, input_df, run_type):
    '''
    Use registered mlflow model in stage based on passed run_type argument to generate score for a features_df
    '''
    # Generate features df from inputs
    features_df = create_features(input_df)

    # Get mlflow_config
    mlflow_config = config['model']['mlflow_config']

    # Get model name and type of model to grab
    model_name = mlflow_config['production_config']['register_model_name']
    model_stage = 'Production' if run_type == 'PROD' else 'Staging'

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])

    # Load the trained model
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    # Predict on a Pandas DataFrame.
    prediction = model.predict(pd.DataFrame(features_df))

    # Set meaning of prediction
    prediction_meaning = f'The model predicted {prediction} which means '
    prediction_meaning = prediction_meaning + 'you is cool' if prediction == 1 else prediction_meaning + 'you is not cool'

    return prediction, prediction_meaning

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    args.add_argument("--run_type", default="DEV")
    args.add_argument("--first_name")
    args.add_argument("--last_initial")
    parsed_args = args.parse_args()

    args_df=pd.DataFrame({'First Name':[parsed_args.first_name], 'Last Name Initial':[parsed_args.last_initial]})
    parsed_config = read_params(parsed_args.config)

    score, meaning = score_input(parsed_config, args_df, parsed_args.run_type)
    print(score, meaning)