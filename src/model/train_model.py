import argparse
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report

from data.preprocess import process_csv

def get_feat_and_target(df,features,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe, features, and target column
    output: two dataframes for x and y 
    """
    X=df[features]
    y=df[[target]]
    return X,y

def train_model(model_config):

    # Load train and test data from config
    train = pd.read_csv(model_config['train_data_path'], sep=",")
    test = pd.read_csv(model_config['test_data_path'], sep=",")

    # Split in to feature and target dataframes based on config
    train_x,train_y=get_feat_and_target(train, model_config['features'], model_config['target'])
    test_x,test_y=get_feat_and_target(test, model_config['features'], model_config['target'])

    # Build and train the machine learning model
    model = RandomForestClassifier(max_depth=model_config['max_depth'],n_estimators=model_config['n_estimators'])
    model.fit(train_x, test_y)

    return model, test_x, test_y

def evaluate_model(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['not cool','cool']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def main(config,input_csv_path, mlflow_tracking_uri, output_model_path):
    # Establish model config dictionary
    model_config = config['model_config']

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Start the MLflow experiment
    mlflow.set_experiment("Coolness Prediction")

    with mlflow.start_run():
        # Track the parameters
        mlflow.log_param("model_config", model_config)

        # Train the model
        model, X_test, y_test = train_model(model_config)

        # Track the model artifacts
        mlflow.sklearn.log_model(model, "coolness_model")

        # Evaluate the model
        test_predictions = model.predict(X_test)
        evaluate_model(y_test,test_predictions,model_config['avg_method'])
        mlflow.log_metric("test_rmse", metrics.mean_squared_error(y_test, test_predictions)

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)

    main(config)
