.PHONY: create_dev_environment install_dependencies start_mlflow

SHELL := /bin/bash

create_dev_environment:
	conda create --name ucool python=3.10 -y

install_dependencies:
	pip install -r requirements.txt

start_mlfow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
