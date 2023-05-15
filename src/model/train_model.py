import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report

def get_feat_and_target(df,features,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe, features, and target column
    output: two dataframes for x and y 
    """
    X=df[features]
    y=df[[target]]
    return X,y

def load_and_prepare_data_for_model(model_config):
    # Load train and test data from config
    train = pd.read_csv(model_config['train_data_path'], sep=",")
    test = pd.read_csv(model_config['test_data_path'], sep=",")

    # Split in to feature and target dataframes based on config
    train_x, train_y = get_feat_and_target(train, model_config['features'], model_config['target'])
    test_x, test_y = get_feat_and_target(test, model_config['features'], model_config['target'])

    return train_x,train_y,test_x,test_y


def train_model(model_config,train_x,train_y):

    # Build and train the machine learning model
    model = RandomForestClassifier(max_depth=model_config['max_depth'],n_estimators=model_config['n_estimators'])
    model.fit(train_x, train_y.values.ravel())

    return model

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

def main(config):
    # Establish model config dictionary
    model_config = config['model']
    mlflow_config = model_config['mlflow_config']

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])

    # Start the MLflow experiment
    mlflow.set_experiment(mlflow_config['experiment_name'])

    with mlflow.start_run():
        # Track the parameters
        mlflow.log_param("model_config", model_config)
        mlflow.log_param("max_depth", model_config["max_depth"])
        mlflow.log_param("n_estimators", model_config["n_estimators"])


        # Prepare data from model
        train_x,train_y,test_x,test_y = load_and_prepare_data_for_model(model_config)

        # Train the model
        model = train_model(model_config, train_x,train_y)

        # Evaluate the model
        test_predictions = model.predict(test_x)
        accuracy,precision,recall,f1score = evaluate_model(
            test_y,test_predictions,model_config['avg_method'])

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        # Track the model artifacts
        signature = infer_signature(train_x, test_predictions)
        mlflow.sklearn.log_model(model, artifact_path="artifacts", signature=signature)

    return train_x,train_y,test_x,test_y, model

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()

    main(read_params(parsed_args.config))
