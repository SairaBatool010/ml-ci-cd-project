import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Enable MLflow auto logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Make sure it matches your MLflow server
mlflow.set_experiment("My Experiment")

with mlflow.start_run():
    # Load dataset
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print("Run logged successfully!")
