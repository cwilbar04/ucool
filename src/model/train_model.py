import argparse
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from preprocess import process_csv

def get_feat_and_target(df,features,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe, features, and target column
    output: two dataframes for x and y 
    """
    X=df[features]
    y=df[[target]]
    return X,y

def train_model(df, config):
    train_data_path = config
    test_data_path

    # Split train and test features 
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    X = data[['Length of First Name', 'Distance']]
    y = data['Coolness']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the machine learning model
    model = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def main(input_csv_path, mlflow_tracking_uri, output_model_path):
    # Read the CSV file and preprocess the data
    data = process_csv(input_csv_path)

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Start the MLflow experiment
    mlflow.set_experiment("Coolness Prediction")

    with mlflow.start_run():
        # Track the parameters
        mlflow.log_param("csv_file_path", input_csv_path)
        mlflow.log_param("output_model_path", output_model_path)
        mlflow.log_param("mlflow_tracking_uri", mlflow_tracking_uri)

        # Train the model
        model, X_test, y_test = train_model(data)

        # Track the model artifacts
        mlflow.sklearn.log_model(model, "coolness_model")

        # Evaluate the model
        test_predictions = model.predict(X_test)
        mlflow.log_metric("test_rmse", metrics.mean_squared_error(y_test, test_predictions))
                          
    #Save the model locally
    save_model(model, output_model_path)

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)
    parser = argparse.ArgumentParser(description='Data preprocessing and machine learning training script')
    parser.add_argument('config', type=str, help='Path to the CSV file')
    parser.add_argument('mlflow_tracking_uri', type=str, help='MLflow tracking URI')
    parser.add_argument('model_filepath', type=str, help='Path to save the trained model as a pickle file')

    args = parser.parse_args()

    main(args.csv_file, args.mlflow_tracking_uri, args.model_filepath)
