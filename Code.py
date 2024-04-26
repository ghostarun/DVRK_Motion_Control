import csv
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pprint import pprint
from math import sqrt
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import control as ctrl
from time import time
import pywt
from scipy.fft import fft, fftfreq
class RLSFilter:
    def __init__(self, n_dim, lambda_=0.99, delta=0.01):
        self.n_dim = n_dim
        self.lambda_ = lambda_
        self.weights = np.zeros(n_dim)
        self.P = np.eye(n_dim) / delta

    def update(self, input_vector, desired_output):
        y = np.dot(self.weights, input_vector)
        e = desired_output - y
        P_x = np.dot(self.P, input_vector)
        g = P_x / (self.lambda_ + np.dot(input_vector, P_x))
        self.weights += g * e
        self.P = (self.P - np.outer(g, P_x)) / self.lambda_
        return y + e

def exponential_moving_average(data, alpha=0.3):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def triple_exponential_moving_average(data, alpha=0.3):
    ema1 = exponential_moving_average(data, alpha)
    ema2 = exponential_moving_average(ema1, alpha)
    ema3 = exponential_moving_average(ema2, alpha)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema

def knn_smooth(data, k=3):
    smoothed_data = np.copy(data)
    for i in range(len(data)):
        indices = range(max(0, i - k // 2), min(len(data), i + k // 2 + 1))
        weights = np.array([1.0 / (abs(i - idx) + 1) for idx in indices])
        smoothed_data[i] = np.average(data[indices], weights=weights)
    return smoothed_data

def wavelet_denoising(data, wavelet='db1', level=None):
    coeff = pywt.wavedec(data, wavelet, mode='symmetric')
    thresholds = [mad(c) * np.sqrt(2 * np.log(len(data))) for c in coeff[1:]]
    new_coeff = [coeff[0]] + [pywt.threshold(c, value=t, mode='soft') for c, t in zip(coeff[1:], thresholds)]
    reconstructed_signal = pywt.waverec(new_coeff, wavelet, mode='symmetric')
    return reconstructed_signal[:len(data)]

def mad(data):
    return np.median(np.abs(data - np.median(data)))

def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())

def process_and_filter_data(paired_datasets, filter_type='rls', alpha=0.3, wavelet='db1', level=None, k=3):
    # Create a new dictionary to store the filtered datasets
    filtered_datasets = {}

    # Define filter methods for each type
    filter_methods = {
        'rls': lambda x, y: RLSFilter(x, y),  # Assuming rls_filter is defined elsewhere to handle arrays
        'ema': lambda data: exponential_moving_average(data, alpha),
        'knn': lambda data: knn_smooth(data, k),
        'wavelet': lambda data: wavelet_denoising(data, wavelet, level)
    }

    # Iterate over all datasets in the dictionary
    for identifier, dataset in paired_datasets.items():
        start = time()
        df = dataset.copy()  # Copy the original dataset to avoid modifying it directly

        measure_columns = ['Position_X_measure', 'Position_Y_measure', 'Position_Z_measure',
                           'Orientation_Z_measure', 'Orientation_Y_measure', 'Orientation_X_measure']
        setpoint_columns = ['Position_X_setpoint', 'Position_Y_setpoint', 'Position_Z_setpoint',
                            'Orientation_Z_setpoint', 'Orientation_Y_setpoint', 'Orientation_X_setpoint']

        # Apply the selected filter method
        for measure_col, setpoint_col in zip(measure_columns, setpoint_columns):
            setpoint_col_filtered = f'{setpoint_col}_filtered'
            if filter_type == 'rls':
                rls_filter_instance = RLSFilter(1)  # Assuming 1D data and RLSFilter class defined properly
                filtered_data = [rls_filter_instance.update([df.at[i, setpoint_col]], df.at[i, measure_col]) for i in df.index]
                df[setpoint_col_filtered] = filtered_data
            else:
                df[setpoint_col_filtered] = filter_methods[filter_type](df[setpoint_col].values)

            # Calculate RMSE before and after filtering for debugging or analysis purposes
            initial_rmse = np.sqrt(np.mean(np.square(df[measure_col] - df[setpoint_col])))
            filtered_rmse = np.sqrt(np.mean(np.square(df[measure_col] - df[setpoint_col_filtered])))
            print(f"{identifier} {setpoint_col} - Initial RMSE: {initial_rmse}, Filtered RMSE: {filtered_rmse}")

        elapsed = time() - start
        print(f"Filtering for {identifier} took {elapsed:.2f} seconds")
        # Store the modified DataFrame back in the new dictionary
        filtered_datasets[identifier] = df

    return filtered_datasets

# Function to process the entire CSV file
def process_csv(file_path):
    positions = []
    orientations = []
    times = []
    first_timestamp = None

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row:  # Check if the row is not empty
                try:
                    # Process the row and extract the data
                    clean_data = row[0].strip().rstrip(',') # Remove whitespace and trailing comma
                    parsed_data = list(map(float, clean_data.split(',')))  # Split and convert to floats
                    position = parsed_data[0:3]
                    orientation = parsed_data[3:6]
                    timestamp = parsed_data[6]
                    
                    # Normalize the first timestamp to zero
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    normalized_timestamp = timestamp - first_timestamp
                    
                    # Store the processed data
                    positions.append(position)
                    orientations.append(orientation)
                    times.append(normalized_timestamp)

                except ValueError as e:
                    # Handle rows that cannot be converted to float
                    print(f"Skipping row due to error: {e}")

    # Create a DataFrame from the extracted data
    processed_data = pd.DataFrame({
        'Position_X': [pos[0] for pos in positions],
        'Position_Y': [pos[1] for pos in positions],
        'Position_Z': [pos[2] for pos in positions],
        'Orientation_Z': [ori[0] for ori in orientations],
        'Orientation_Y': [ori[1] for ori in orientations],
        'Orientation_X': [ori[2] for ori in orientations],
        'Time': times
    })

    return processed_data

def create_datasets(data_directory):
    paired_datasets = {}

    # Get a list of all 'measure' CSV files
    measure_files = glob.glob(os.path.join(data_directory, 'measure*.csv'))

    for measure_file in measure_files:
        # Get the base name and construct the corresponding 'setpoint' filename
        base_name = os.path.basename(measure_file)  # Gets the filename without the path
        identifier = base_name.replace('measure', '').replace('.csv', '')  # Isolate the identifier
        setpoint_file = os.path.join(data_directory, 'setpoint' + identifier + '.csv')

        # Check if the corresponding 'setpoint' file exists
        if os.path.exists(setpoint_file):
            # Process and merge the 'measure' and 'setpoint' data
            print(f"Processing {measure_file} and {setpoint_file}")
            measure_data = process_csv(measure_file)
            setpoint_data = process_csv(setpoint_file)
            merged_data = pd.merge(measure_data, setpoint_data, on='Time', how='inner', suffixes=('_measure', '_setpoint'))
            columns = [col for col in merged_data.columns if col != 'Time'] + ['Time']
            merged_data = merged_data[columns]
            merged_data.name = identifier  # Assign the common identifier to the merged DataFrame
            
            # Store the merged DataFrame in the dictionary with its identifier
            paired_datasets[identifier] = merged_data
        else:
            print(f"No matching setpoint file for {measure_file}")

    return paired_datasets

def plot_rmse_over_time(paired_datasets, filtered: bool = False):
    for identifier, dataset in paired_datasets.items():
        # Prepare the data directory for saving plots
        os.makedirs('Plots', exist_ok=True)

        # Initialize the first figure for line graphs
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        fig.suptitle(f'RMSE over Time for {identifier}', fontsize=16)

        position_rmses = []
        orientation_rmses = []

        # Plot settings for line graphs
        position_colors = ['blue', 'green', 'red']
        orientation_colors = ['crimson', 'mediumseagreen', 'slateblue']

        # For each axis in position (X, Y, Z)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            measure_col = f'Position_{axis}_measure'
            if filtered:
                setpoint_col = f'Position_{axis}_setpoint_filtered'
            else:
                setpoint_col = f'Position_{axis}_setpoint'
            rmse = np.sqrt((dataset[measure_col] - dataset[setpoint_col]) ** 2)
            position_rmses.append(rmse)
            axes[0].plot(dataset['Time'], rmse, label=f'Position {axis}', color=position_colors[i])
        
        axes[0].set_title('Position RMSE')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('RMSE')
        axes[0].legend()

        # For each axis in orientation (Z, Y, X)
        for i, axis in enumerate(['Z', 'Y', 'X']):
            measure_col = f'Orientation_{axis}_measure'
            if filtered:
                setpoint_col = f'Orientation_{axis}_setpoint_filtered'
            else:
                setpoint_col = f'Orientation_{axis}_setpoint'
            rmse = np.sqrt((dataset[measure_col] - dataset[setpoint_col]) ** 2)
            orientation_rmses.append(rmse)
            axes[1].plot(dataset['Time'], rmse, label=f'Orientation {axis}', color=orientation_colors[i])

        axes[1].set_title('Orientation RMSE')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('RMSE')
        axes[1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if filtered:
            plt.savefig(f'Plots/{identifier}_RMSE_filtered.png')
        else:
            plt.savefig(f'Plots/{identifier}_RMSE.png')
        
        plt.close(fig)  # Close the figure to free up memory

        # Initialize the second figure for mean RMSE bar graphs
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.suptitle(f'Mean RMSE for {identifier}', fontsize=16)

        # Plot mean RMSE for Position
        mean_position_rmses = [np.mean(rmse) for rmse in position_rmses]
        axes[0].bar(['X', 'Y', 'Z'], mean_position_rmses, color=position_colors)
        axes[0].set_title('Mean Position RMSE')
        axes[0].set_ylabel('RMSE')

        # Plot mean RMSE for Orientation
        mean_orientation_rmses = [np.mean(rmse) for rmse in orientation_rmses]
        axes[1].bar(['Z', 'Y', 'X'], mean_orientation_rmses, color=orientation_colors)
        axes[1].set_title('Mean Orientation RMSE')
        axes[1].set_ylabel('RMSE')

        plt.tight_layout()
        if filtered:
            plt.savefig(f'Plots/{identifier}_RMSE_mean_filtered.png')
        else:
            plt.savefig(f'Plots/{identifier}_RMSE_mean.png')
        plt.close(fig)  # Close the figure to free up memory

def plot_rmse_comparisons(paired_datasets, filtered: bool = False):
    # Ensure the Plots directory exists
    os.makedirs('Plots', exist_ok=True)

    for identifier, dataset in paired_datasets.items():
        # Extract position and orientation data
        positions = [dataset[f'Position_{axis}_setpoint_filtered' if filtered  else f'Position_{axis}_setpoint'] - dataset[f'Position_{axis}_measure'] for axis in ['X', 'Y', 'Z']]
        orientations = [dataset[f'Orientation_{axis}_setpoint_filtered' if filtered  else f'Orientation_{axis}_setpoint'] - dataset[f'Orientation_{axis}_measure'] for axis in ['Z', 'Y', 'X']]
        timestamps = dataset['Time']

        # Setup figure for position RMSEs
        pos_figure, pos_axes = plt.subplots(1, 3, figsize=(20, 5))
        pos_figure.suptitle(f"Position RMSE Comparisons for {identifier}", fontsize=16, fontweight='bold')

        # Position comparisons
        for ax, diff, axis_label in zip(pos_axes, positions, ['X', 'Y', 'Z']):
            rmse = np.sqrt(diff ** 2)
            max_rmse = np.max(rmse)
            ax.plot(timestamps, rmse, label=f'{axis_label} Axis', color='dodgerblue', linewidth=2)
            ax.set_title(f"{axis_label} Position RMSE")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("RMSE (units)")
            ax.legend()
            ax.grid(True)
            ax.set_facecolor('whitesmoke')
            ax.set_ylim([-max_rmse, max_rmse])  # Symmetric y-limits around zero

        plt.tight_layout()
        if filtered:
            plt.savefig(f'Plots/{identifier}_Position_RMSE_filtered.png')
        else:
            plt.savefig(f'Plots/{identifier}_Position_RMSE.png')
        plt.close(pos_figure)

        # Setup figure for orientation RMSEs
        orient_figure, orient_axes = plt.subplots(1, 3, figsize=(20, 5))
        orient_figure.suptitle(f"Orientation RMSE Comparisons for {identifier}", fontsize=16, fontweight='bold')

        # Orientation comparisons
        for ax, diff, axis_label in zip(orient_axes, orientations, ['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']):
            rmse = np.sqrt(diff ** 2)
            max_rmse = np.max(rmse)
            ax.plot(timestamps, rmse, label=f'{axis_label}', color='tomato', linewidth=2)
            ax.set_title(f"{axis_label} RMSE")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("RMSE (radians)")
            ax.legend()
            ax.grid(True)
            ax.set_facecolor('whitesmoke')
            ax.set_ylim([-max_rmse, max_rmse])  # Symmetric y-limits around zero

        plt.tight_layout()
        if filtered:
            plt.savefig(f'Plots/{identifier}_Orientation_RMSE_filtered.png')
        else:
            plt.savefig(f'Plots/{identifier}_Orientation_RMSE.png')
        plt.close(orient_figure)


def plot_trajectories(paired_datasets, filtered: bool = False):
    # Ensure the Plots directory exists
    os.makedirs('Plots', exist_ok=True)

    for identifier, dataset in paired_datasets.items():
        # Initialize the figure for 3D trajectory plots
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set the background color for the plot
        ax.set_facecolor('whitesmoke')  # Light grey background for better contrast

        # Disable the grid background
        ax.xaxis.pane.fill = False  # Disable the background pane for x-axis
        ax.yaxis.pane.fill = False  # Disable the background pane for y-axis
        ax.zaxis.pane.fill = False  # Disable the background pane for z-axis

        ax.grid(True, linestyle='--', linewidth=1, color='grey', alpha=0.5)  # Grey dashed grid lines

        # Plot the setpoint trajectory
        if filtered:
            ax.plot(dataset['Position_X_setpoint_filtered'], dataset['Position_Y_setpoint_filtered'], dataset['Position_Z_setpoint_filtered'],
                    label='Filtered Setpoint Trajectory', color='dodgerblue', linewidth=2)
        else:
            ax.plot(dataset['Position_X_setpoint'], dataset['Position_Y_setpoint'], dataset['Position_Z_setpoint'],
                    label='Setpoint Trajectory', color='dodgerblue', linewidth=2)

        # Plot the measured trajectory
        ax.plot(dataset['Position_X_measure'], dataset['Position_Y_measure'], dataset['Position_Z_measure'],
                label='Measured Trajectory', color='springgreen', linewidth=2, alpha=0.6)

        # Set plot labels and title
        ax.set_title(f'Trajectory Comparison for {identifier}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()

        plt.tight_layout()
        if filtered:
            plt.savefig(f'Plots/{identifier}_Filtered_Trajectory.png')
        else:
            plt.savefig(f'Plots/{identifier}_Trajectory.png')
        plt.close(fig)


from scipy.fft import fft

def fft_on_dataset(filtered_datasets, datasets):
    """
    Perform FFT on measurement and setpoint data of paired datasets.

    Parameters:
    - filtered_datasets: A dictionary of DataFrames, with each key as an identifier
                       and each value as a DataFrame containing the trajectory data.

    Returns:
    - fft_datasets: A dictionary containing the FFT results for each dataset.
    """

    # Dictionary to store the FFT results
    fft_datasets = {}

    for identifier, dataset in filtered_datasets.items():
        # Copy the dataset to avoid modifying the original data
        df = dataset.copy()

        measure_columns = ['Position_X_measure', 'Position_Y_measure', 'Position_Z_measure',
                           'Orientation_Z_measure', 'Orientation_Y_measure', 'Orientation_X_measure']
        setpoint_columns = ['Position_X_setpoint', 'Position_Y_setpoint', 'Position_Z_setpoint',
                            'Orientation_Z_setpoint', 'Orientation_Y_setpoint', 'Orientation_X_setpoint']
        # positions = [
        #     dataset[f'Position_{axis}_setpoint_filtered' if filtered else f'Position_{axis}_setpoint'] - dataset[
        #         f'Position_{axis}_measure'] for axis in ['X', 'Y', 'Z']]
        # orientations = [
        #     dataset[f'Orientation_{axis}_setpoint_filtered' if filtered else f'Orientation_{axis}_setpoint'] - dataset[
        #         f'Orientation_{axis}_measure'] for axis in ['Z', 'Y', 'X']]
        # timestamps = dataset['Time']

        # Dictionary to store FFT results for the current dataset
        fft_results = {}

        # Perform FFT for each measurement and setpoint column
        for measure_col, setpoint_col in zip(measure_columns, setpoint_columns):
            # Perform FFT
            fft_measure = fft(df[measure_col].values)
            fft_setpoint = fft(df[setpoint_col].values)

            # Store the results
            fft_results[f'{measure_col}_fft'] = fft_measure
            fft_results[f'{setpoint_col}_fft'] = fft_setpoint

        # Add the results for the current dataset to the main dictionary
        fft_datasets[identifier] = fft_results

    return fft_datasets

# Example usage:
data_directory = 'Data'
datasets = create_datasets(data_directory)

# plot_rmse_over_time(datasets, filtered=False)
# plot_rmse_comparisons(datasets, filtered=False)
# plot_trajectories(datasets, filtered=False)

filtered_datasets = process_and_filter_data(datasets, filter_type='rls')

# plot_rmse_over_time(filtered_datasets, filtered=True)
# plot_trajectories(filtered_datasets, filtered=True)
# plot_rmse_comparisons(filtered_datasets, filtered=True)


fft_results = fft_on_dataset(filtered_datasets, datasets)



