.PHONY: create_environment install_dependencies add_pythonpath setup_env start_mlflow

SHELL := /bin/bash

create_environment:
	conda create --name ucool python=3.10 -y

add_pythonpath:
	export PYTHONPATH="$(PWD)/src"

setup_env:
	$(MAKE) add_pythonpath && \
	$(MAKE) create_environment

install_dependencies:
	pip install -r requirements-dev.txt

start_mlfow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
