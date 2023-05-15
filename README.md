# ucool

Demo project for setting up elements of productionalization

## Getting Started
1. Create Conda Environment
```cmd
make create_dev_environment
```
2. Activate Conda Environment
```cmd
conda activate ucool
```
3. Install Development Requirements Pacakges
```cmd
pip install -r requirements-dev.txt
```
4. Setup PYTHONPATH to recognize source code
```cmd
export PYTHONPATH="$PWD/src"
```
5. Generate Demo Data
```cmd
python ./data_generator/generate_data.py 10000 ./data_generator/demo.csv
```
6. Start MLFLOW server
```cmd
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
```

# Got started in class
made an additional change