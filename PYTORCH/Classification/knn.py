import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data(csv_file):
    """Load data from a CSV file."""
    df = pd.read_csv(csv_file)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    return X, y

def train_and_evaluate(rank, world_size, config):
    """Train and evaluate KNN model."""
    # Initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Load the full dataset
    X, y = load_data("dataset1.csv")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Partition the data for each node
    partition_size = len(X_train) // world_size
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != world_size - 1 else len(X_train)

    X_train_partition = X_train[start_idx:end_idx]
    y_train_partition = y_train[start_idx:end_idx]

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=config['n_neighbors'], n_jobs=config['n_jobs'])

    # Record the start time
    start_time = time.time()
    
    # Train the model
    knn.fit(X_train_partition, y_train_partition)

    # Make predictions and evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Record the end time
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Rank {rank}: Accuracy = {accuracy}")
    print(f"Rank {rank}: Training Time = {training_time:.4f} seconds")

    # Clean up
    dist.destroy_process_group()

def run_experiment(config, world_size):
    # Start multiprocessing with the specified number of processes
    mp.spawn(train_and_evaluate, args=(world_size, config), nprocs=1, join=True)

if __name__ == '__main__':
    # Define configuration for KNN
    config = {
        "n_neighbors": 5,  # Number of neighbors for KNN
        "n_jobs": 4        # Use 4 CPUs for each process (if available)
    }
    world_size = 3  # Number of nodes

    start = time.time()

    # Run the experiment using PyTorch multiprocessing
    run_experiment(config, world_size)

    end = time.time()
    print("Model Training Fit Time:", end - start)
