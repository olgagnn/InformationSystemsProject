import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def load_data(csv_file):
    """Load data from a CSV file."""
    df = pd.read_csv(csv_file)
    X = df.drop('target', axis=1).values  # Features (assuming the target column is named 'target')
    y = df['target'].values  # Target labels
    return X, y

def train_and_evaluate(rank, world_size, config):
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Load the full dataset
    X, y = load_data("dataset1.csv")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Partition the training data
    partition_size = len(X_train) // world_size
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != world_size - 1 else len(X_train)

    X_train_partition = X_train[start_idx:end_idx]
    y_train_partition = y_train[start_idx:end_idx]

    # Train the model using XGBoost
    clf = XGBClassifier(**config)
    clf.fit(X_train_partition, y_train_partition)

    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy for each process
    print(f"Rank {rank} Accuracy: {accuracy}")

    # Clean up
    dist.destroy_process_group()

def run_experiment(config, world_size):
    # Start multiprocessing with the specified number of processes
    mp.spawn(train_and_evaluate, args=(world_size, config), nprocs=1, join=True)

if __name__ == '__main__':
    start = time.time()

    # Define configuration for the experiment
    config = {'n_jobs': 4}  # Use 4 CPU cores for each process (trial)
    world_size = 3  # Number of nodes

    # Run the experiment using PyTorch multiprocessing
    run_experiment(config, world_size)

    end = time.time()
    print("Model Training Fit Time:", end - start)
