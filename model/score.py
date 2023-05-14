import mlflow
import pandas as pd

logged_model = 'runs:/113e3aceb54545a19b29b7f29e0ed8df/coolness_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.

loaded_model.predict(pd.DataFrame(data))