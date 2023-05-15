import mlflow
from mlflow import MlflowClient

def move_model_stage(config, run_type='TEST', model_version=None):
    """
    Change stage of registered model depending on run type
    """
    # Get MLFLow config
    mlflow_config = config['model']['mlflow_config']
    mlflow.set_tracking_uri(mlflow_config['remote_server_uri'])

    #Stage & Version to move to depends on run type
    if run_type == 'PROD':
        stage = 'Production'
        model_version = mlflow_config['production_config']['production_version']
    else:
        stage = 'Staging'
        if not model_version:
            raise Exception('Please provide version number for non-prod run type')
        
    model_name = mlflow_config['production_config']['register_model_name']

    # Move immediately into Staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage
    )

    message = f'Moved {model_name} with version {model_version} to {stage}'

    return message

if __name__ == '__main__':
    import argparse
    from helpers import read_params

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    args.add_argument("--run_type", default="DEV")
    args.add_argument("--model_version", default = None)
    parsed_args = args.parse_args()

    parsed_config = read_params(parsed_args.config)
    version = int(parsed_args.model_version) if parsed_args.model_version else None

    result = move_model_stage(parsed_config, parsed_args.run_type, version)
