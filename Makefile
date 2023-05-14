.PHONY: create_environment install_dependencies add_pythonpath setup_env start_mlfow

create_environment:
	conda create --name ucool python=3.10

install_dependencies:
	conda activate ucool && \
	pip install -r requirements-dev.txt

add_pythonpath:
	export PYTHONPATH="$(PWD)/src"

setup_env:
	$(MAKE) create_environment
	$(MAKE) install_dependencies
	$(MAKE) add_pythonpath

start_mlfow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
