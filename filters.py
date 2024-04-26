import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class RLSFilter:
    def __init__(self, filter_order, lambda_, delta):
        self.n = filter_order
        self.lambda_ = lambda_
        self.weights = np.zeros(self.n)
        self.P = np.eye(self.n) / delta

    def update(self, input_vector, desired_output):
        filter_output = np.dot(self.weights, input_vector)
        estimation_error = desired_output - filter_output
        Pi = np.dot(self.P, input_vector)
        L = Pi / (self.lambda_ + np.dot(input_vector, Pi))
        self.weights += L * estimation_error
        self.P = (self.P - np.outer(L, Pi)) / self.lambda_
        return filter_output

def apply_rls_to_datasets(paired_datasets):
    os.makedirs('Plots', exist_ok=True)
    for identifier, df in paired_datasets.items():
        # Check and print columns
        print(f"Columns in {identifier}: {df.columns.tolist()}")  # Debugging line

        filter_order = 4
        lambda_ = 0.99
        delta = 0.01
        filters = {axis: RLSFilter(filter_order, lambda_, delta) for axis in ['Position_X', 'Position_Y', 'Position_Z']}
        
        # Prepare DataFrame to store filtered results
        for axis in ['Position_X', 'Position_Y', 'Position_Z']:
            df[f'{axis}_filtered'] = np.nan
        
        for i in range(filter_order, len(df)):
            for axis in ['Position_X', 'Position_Y', 'Position_Z']:
                input_vector = df[axis].iloc[i-filter_order:i].values
                desired_output = df[axis].iloc[i]
                df.at[i, f'{axis}_filtered'] = filters[axis].update(input_vector, desired_output)

    return paired_datasets

# Assuming 'datasets' is your dictionary of paired dataframes
datasets = {
    'example_dataset': pd.DataFrame({
        'Time': np.linspace(0, 10, 100),
        'Position_X': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        'Position_Y': np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        'Position_Z': np.sin(np.linspace(0, 10, 100)) * np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
    })
}
filtered_datasets = apply_rls_to_datasets(datasets)
print(filtered_datasets)
