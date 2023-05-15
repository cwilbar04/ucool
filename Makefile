.PHONY: create_dev_environment install_dependencies start_mlflow code_test lint all_tests install generate_demo_data

SHELL := /bin/bash

create_dev_environment:
	conda create --name ucool python=3.10 -y

start_mlfow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

code_test:
	cd tests
	python -m pytest -vv

lint:
	pylint --disable=R,C src tests

generate_demo_data:
	python ./data_generator/generate_data.py --num_rows=10000 --output_filepath=./data_generator/demo.csv