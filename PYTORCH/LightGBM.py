import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Function to load data
def load_data(csv_file):
    """Load data from a CSV file."""
    df = pd.read_csv(csv_file)
    X = df.drop('target', axis=1).values  # Features 
    y = df['target'].values  # Target labels
    return X, y

# Function to train and evaluate the model using LightGBM
def train_and_evaluate(rank, world_size, config):
    # Initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Load the full dataset (each node will get a different split)
    X, y = load_data("dataset1.csv")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Partition the data for each node
    partition_size = len(X_train) // world_size
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != world_size - 1 else len(X_train)

    X_train_partition = X_train[start_idx:end_idx]
    y_train_partition = y_train[start_idx:end_idx]

    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train_partition, label=y_train_partition)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Train the model
    clf = lgb.train(config, train_data, valid_sets=[test_data])

    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    y_pred = [round(value) for value in y_pred]  # Convert probabilities to class labels

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Process {rank}: Accuracy = {accuracy}")

    # Clean up
    dist.destroy_process_group()

    return {"accuracy": accuracy}

# Function to run the experiment
def run_experiment(world_size, config):
    # Use PyTorch multiprocessing to parallelize the training
    mp.spawn(train_and_evaluate, args=(world_size, config), nprocs=1, join=True)

if __name__ == '__main__':
    # Define configuration for LightGBM
    config = {
        "n_jobs": 4  # Use 4 CPUs for each process 
    }

    world_size = 3  # Number of nodes

    start = time.time()

    # Run the experiment
    run_experiment(world_size, config)

    end = time.time()
    print("Model Training Fit Time:", end - start)
