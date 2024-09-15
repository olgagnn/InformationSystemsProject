import os
import time
import ray
from ray import tune
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Ray Initialization
ray.init(address='auto')

# Load data from CSV
def load_data(csv_file):
    """Load data from a CSV file."""
    df = pd.read_csv(csv_file)
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target labels
    return X, y

# Load Data and store in Ray object store
X, y = load_data("dataset1.csv")  
X_id = ray.put(X)
y_id = ray.put(y)

# Training function for KNN
def train_knn(config):
    # Get data from Ray object store
    X = ray.get(X_id)
    y = ray.get(y_id)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN using default parameters
    knn = KNeighborsClassifier(n_neighbors=config.get("n_neighbors", 5))  # Default n_neighbors=5
    knn.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
  
    return {"accuracy": accuracy}

# Absolute path for saving our results
absolute_path = os.path.abspath('./ray_results')

# Ray Tune
start_time = time.time()

analysis = tune.run(
    train_knn,
    config={"n_neighbors": 5}, 
    num_samples=2,  # Run 2 trials
    resources_per_trial={"cpu": 4},  # Allocate 4 CPUs per trial
    storage_path=absolute_path  # Save results in the specified directory
)

end_time = time.time()
print("KNN Training Time:", end_time - start_time)

# Get config and accuracy
config = analysis.get_best_config(metric="accuracy", mode="max")
accuracy = analysis.get_best_trial(metric="accuracy", mode="max").last_result["accuracy"]

print("Configuration:", best_config)
print("Accuracy:", best_accuracy)
