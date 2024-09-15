import os
import time
import ray
from ray import tune
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ray Initialization
ray.init(address='auto')

def load_data(csv_file):
    """Load data from a CSV file."""
    df = pd.read_csv(csv_file)
    X = df.drop('target', axis=1)  # Features 
    y = df['target']  # Target labels
    return X, y

# Load Data and store in Ray object store
X, y = load_data("dataset3.csv")
X_id = ray.put(X)
y_id = ray.put(y)

def train_and_evaluate(config):
    # Get the data from Ray object store
    X_data = ray.get(X_id)
    y_data = ray.get(y_id)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Initialize and train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=config["n_estimators"], random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"accuracy": accuracy}

# Absolute path for saving our results
absolute_path = os.path.abspath('./ray_results')

# Ray Tune
start = time.time()
analysis = tune.run(
    train_and_evaluate,
    config={
        "n_estimators": 10 
    },
    num_samples=2, 
    resources_per_trial={"cpu": 4},
    storage_path=absolute_path,  # Use absolute path
)
end = time.time()
print("Model Training Fit Time:", end - start)

# Print configuration and accuracy
config = analysis.get_best_config(metric="accuracy", mode="max")
accuracy = analysis.get_best_trial(metric="accuracy", mode="max").last_result["accuracy"]
print("Configuration:", best_config)
print("Accuracy:", best_accuracy)
