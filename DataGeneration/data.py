import os
import argparse
import pandas as pd
from sklearn.datasets import make_classification

def classification_generate_data(num_samples, num_features):
    # Temp and final output files
    temp_file = 'data_temp.csv'
    output_file = 'dataset3.csv'
    metadata_file = 'dataset3.csv.meta'

    # Remove temp and output files if they already exist
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    # Define chunk size (because of our system's memory constraints)
    chunk_size = 3 * (10**5)

    # If the number of samples exceeds the chunk size, generate data in chunks
    if num_samples > chunk_size:
        num_chunks = num_samples // chunk_size
        remainder = num_samples % chunk_size

        for i in range(num_chunks):
            print(f"Creating chunk {i+1} of {num_chunks + (1 if remainder > 0 else 0)}")
            X, y = make_classification(
                n_samples=chunk_size,
                n_features=num_features,
                n_informative=num_features // 2,  
                n_redundant=num_features // 2,    
                n_classes=2  
            )

            # Convert to DataFrame for CSV output
            df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(num_features)])
            df['target'] = y

            # Append chunk to the CSV file
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        if remainder > 0:
            print(f"Creating remainder chunk {num_chunks+1} of {num_chunks+1}")
            X, y = make_classification(
                n_samples=remainder,
                n_features=num_features,
                n_informative=num_features // 2,  
                n_redundant=num_features // 2,    
                n_classes=2  
            )

            df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(num_features)])
            df['target'] = y

            df.to_csv(output_file, mode='a', header=False, index=False)

        # Get the size of the output CSV file in MB
        dataset_size_mb = os.path.getsize(output_file) / 10**6

        # Write metadata file
        with open(metadata_file, 'w') as f:
            f.write(f'num_samples,{num_samples}\n')
            f.write(f'num_features,{num_features}\n')
            f.write(f'dataset_size_mb,{dataset_size_mb:.2f}\n')

        print(f"Dataset size is: {dataset_size_mb:.2f} MB")
        return num_features, num_samples

    else:
        # If samples are within chunk size, generate all at once
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features // 2,  # Adjust as needed
            n_redundant=num_features // 2,    # Adjust as needed
            n_classes=2  # Default number of classes is 2
        )

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
        df['target'] = y
        df.to_csv(output_file, index=False)

        # Get the size of the output CSV file in MB
        dataset_size_mb = os.path.getsize(output_file) / 10**6

        # Write metadata file
        with open(metadata_file, 'w') as f:
            f.write(f'num_samples,{num_samples}\n')
            f.write(f'num_features,{num_features}\n')
            f.write(f'dataset_size_mb,{dataset_size_mb:.2f}\n')

        print(f"Dataset size is: {dataset_size_mb:.2f} MB")
        return num_features, num_samples

def main():
    parser = argparse.ArgumentParser(description='Generate Classification Data.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--num_features', type=int, default=20, help='Number of features')

    args = parser.parse_args()

    # Call the function with command-line arguments
    classification_generate_data(num_samples=args.num_samples, num_features=args.num_features)

if __name__ == '__main__':
    main()
