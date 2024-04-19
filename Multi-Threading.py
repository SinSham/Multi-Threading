import numpy as np
import random 
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool


def multiply_matrices(args):
    matrix1, matrix2 = args
    return np.dot(matrix1, matrix2)


def matrMultiplication(num_processes):
    
    mat1 = np.random.rand(5000, 5000)
    mat2 = np.random.rand(5000, 5000)
    split_matrix1 = np.array_split(mat1, num_processes, axis=0)
    split_matrix2 = np.array_split(mat2, num_processes, axis=1)

    pool = Pool(processes=num_processes)

    start_time = time.time()

    # Perform matrix multiplication using multiprocessing
    results = pool.map(multiply_matrices, zip(split_matrix1, split_matrix2))

    # Concatenate results
    result = np.concatenate(results, axis=1)

    end_time = time.time()

    # Calculate the time taken
    elapsed_time = end_time - start_time
    return elapsed_time

if __name__ == "__main__":
    num_processes_values = [1, 2, 4, 8]  # Test different number of processes

    # Measure time taken for matrix multiplication with different number of processes
    times = [matrMultiplication(num_processes) for num_processes in num_processes_values]

    # Plot the comparison
    plt.plot(num_processes_values, times, marker='o')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken for Matrix Multiplication with Different Number of Processes')
    plt.grid(True)
    plt.show()