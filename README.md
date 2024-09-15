# Python Scaling Frameworks for Big Data Analysis and Machine Learning: A Comparison of Ray and PyTorch

Team project for the course 'Information-Systems-Analysis-and-Design' during the 9th semester at NTUA in the School of Electrical and Computer Engineering under the supervision of Professor Dimitrios Tsoumakos.

## Team Members

- **Giannouchou Olga**
- **Aikaterini Stavrou**

## Introduction

Τhis project compares Ray and PyTorch, two well-known Python scaling frameworks. Open-source frameworks like PyTorch and Ray make it easier to run Python code in parallel for distributed computing applications. In this comparison, their performance, scalability, and efficiency in managing massive data analytics and machine learning workloads are all intended to be extensively assessed.

### Background
Python has emerged as a popular programming language for data analysis and machine learning due to its simplicity, versatility, and extensive ecosystem of libraries and frameworks. However, as datasets continue to grow in size and complexity, traditional single-machine solutions become inadequate for processing and analyzing such data efficiently. This has led to the development of distributed computing frameworks like Ray and PyTorch, which enable parallel execution across multiple nodes or cores, thereby allowing users to scale their data analysis and ML workflows to handle large-scale datasets.

## Project Objectives

1. **Installation and Setup:** Successfully install and configure Ray and PyTorch frameworks on local or cloud-based resources.
2. **Data Loading and Preprocessing:** Generate or acquire real-world datasets and load them into Ray and PyTorch for analysis. Perform necessary preprocessing steps to prepare the data for analysis.
3. **Performance Measurement:** Develop a suite of Python scripts to benchmark the performance of Ray and PyTorch across various data processing and ML tasks. Evaluate their scalability and efficiency under different workload scenarios.
4. **Comparison Analysis:** Analyze and compare the performance, scalability, and ease of use of Ray and PyTorch based on the results obtained from the performance measurement phase. Identify the strengths and weaknesses of each framework for different types of tasks and workloads.

## Project Steps

To run the Python scripts, follow these steps:

1. **Installation and Setup of Virtual Machines:**
   - Using Okeanos-documentation, create and set up your virtual machines. In this project, we used a total of 3 VMs with 8 gigabytes of RAM, 30 gigabytes of disk storage, and 4 CPUs each (1 master and 2 workers).
   - Create a private network with the three VMs.

2. **Framework Installation:**
   - Install Ray and PyTorch frameworks.
   - Install all necessary libraries.

3. **Data Loading and Preprocessing:**
   - Generate test data using the `data.py` function:
     ```
     python3 data.py --num_samples <num_samples> --num_features <features>
     ```

4. **Classification:**
   - For PyTorch:
     - Run the `torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="<IP>" --master_port=<port> <script_name>` command to initiate the PyTorch distributed cluster.
     - Connect to the worker VMs and run the script using `torchrun --nproc_per_node=1 --nnodes=3 --node_rank=<n> --master_addr="<IP>" --master_port=<port> <script_name>`.

   - For Ray:
     - Initiate the cluster with `ray start --head`.
     - Connect to the cluster with a worker node using `ray start --address='ip_address:port'`.
     - Run the Ray `.py` file.

## References

- [Ray Documentation](https://docs.ray.io/)
- [PyTorch Documentation](https://pytorch.org/)
