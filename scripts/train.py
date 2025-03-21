import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

print("🚀 Starting train.py execution...")  

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

mlflow.set_experiment("My Experiment")
print(f"✅ MLflow experiment set: {mlflow.get_experiment_by_name('My Experiment')}")

with mlflow.start_run():
    print("📊 Loading dataset...")
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    print("🛠️ Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("📌 Logging parameters...")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    print("💾 Logging model...")
    mlflow.sklearn.log_model(model, "model")

    print("✅ Run logged successfully!")

print("🎉 End of script.")  
