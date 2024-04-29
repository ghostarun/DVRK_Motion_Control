from Code import *
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import os


def fft_analysis_and_visualization(paired_datasets):
    os.makedirs('Plots/fft', exist_ok=True)
    for identifier, dataset in paired_datasets.items():
        # Initialize the figure for 3D trajectory plots
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Background and grid setup
        ax.set_facecolor('whitesmoke')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linestyle='--', linewidth=1, color='grey', alpha=0.5)

        # Extract trajectories
        traj_gt = np.stack((dataset['Position_X_setpoint'], dataset['Position_Y_setpoint'], dataset['Position_Z_setpoint']), axis=-1)
        x_gt =  dataset['Position_X_setpoint']
        y_gt =  dataset['Position_Y_setpoint']
        z_gt =  dataset['Position_Z_setpoint']
        traj_filtered = np.stack((dataset['Position_X_setpoint_filtered'], dataset['Position_Y_setpoint_filtered'], dataset['Position_Z_setpoint_filtered']), axis=-1)
        x_filt = dataset['Position_X_setpoint_filtered']
        y_filt = dataset['Position_Y_setpoint_filtered']
        z_filt = dataset['Position_Z_setpoint_filtered']


        # Calculate FFT of both raw and filtered data
        fft_gt = fft(traj_gt)
        fft_filtered = fft(traj_filtered)

        # Compute the spectrum difference
        fft_difference = fft_gt - fft_filtered

        # Calculate the Inverse FFT of the difference to get the time-domain noise signal
        noise_signal = ifft(fft_difference).real  # Taking the real part as the output should be real

        # Time vector for plotting
        time = np.linspace(0, len(traj_gt), num=len(traj_gt), endpoint=False)

        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

        # Plot raw and filtered data
        ax1.plot(time, traj_gt, label='Raw Data', color='gray')
        ax1.plot(time, traj_filtered, label='Filtered Data', color='green')
        ax1.set_title('Raw vs. Filtered Trajectories')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        handles, labels = ax1.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))  # removes duplicates
        ax1.legend(legend_dict.values(), legend_dict.keys())

        # Plot the frequency spectra
        ax2.plot(np.abs(fft_gt), label='FFT of Raw Data', color='red')
        ax2.plot(np.abs(fft_filtered), label='FFT of Filtered Data', color='blue')
        ax2.set_title('Frequency Spectra Comparison')
        ax2.set_xlabel('Frequency Index')
        ax2.set_ylabel('Magnitude')
        handles, labels = ax2.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))  # removes duplicates
        ax2.legend(legend_dict.values(), legend_dict.keys())

        # Plot the noise signal
        ax3.plot(time, noise_signal, label='Noise Signal (Removed by Filtering)', color='purple')
        ax3.set_title('Noise Removed Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Amplitude of Noise')
        handles, labels = ax3.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))  # removes duplicates
        ax3.legend(legend_dict.values(), legend_dict.keys())

        plt.tight_layout()
        plt.savefig(f'Plots/fft/{identifier}_FFT_Analysis.png')
        plt.close(fig)

def track_and_plot_features(paired_datasets, filtered: bool = False):
    os.makedirs('Plots', exist_ok=True)

    def magnitude(x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)

    for identifier, dataset in paired_datasets.items():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('whitesmoke')
        ax.grid(True, linestyle='--', linewidth=1, color='grey', alpha=0.5)


        x_traj = dataset['Position_X_setpoint_filtered'] if filtered else dataset['Position_X_setpoint']
        y_traj = dataset['Position_Y_setpoint_filtered'] if filtered else dataset['Position_Y_setpoint']
        z_traj = dataset['Position_Z_setpoint_filtered'] if filtered else dataset['Position_Z_setpoint']
        x_rot = dataset['Orientation_X_setpoint_filtered'] if filtered else dataset['Orientation_X_setpoint']
        y_rot = dataset['Orientation_Y_setpoint_filtered'] if filtered else dataset['Orientation_Y_setpoint']
        z_rot = dataset['Orientation_Z_setpoint_filtered'] if filtered else dataset['Orientation_Z_setpoint']

        mag_traj = magnitude(x_traj, y_traj, z_traj)
        gradient_traj = np.gradient(mag_traj)
        mag_rot = magnitude(x_rot, y_rot, z_rot)
        gradient_rotx = np.gradient(x_rot)
        gradient_roty = np.gradient(y_rot)
        gradient_rotz = np.gradient(z_rot)
        gradient_rot_mag = magnitude(gradient_rotx, gradient_roty, gradient_rotz)

        ## Clustering trajectory points to find regions of similar behavior
        # kmeans = KMeans(n_clusters=3, n_init='auto')
        # labels = kmeans.fit_predict(
        #     np.stack((dataset['Position_X_setpoint'], dataset['Position_Y_setpoint'], dataset['Position_Z_setpoint']),
        #              axis=-1))
        # colors = ['red', 'green', 'blue']  # Cluster colors
        # ax.plot(x_traj, y_traj, z_traj, label='Setpoint Trajectory', color='dodgerblue', linewidth=2)
        # ax.scatter(dataset['Position_X_setpoint'], dataset['Position_Y_setpoint'], dataset['Position_Z_setpoint'],
        #            c=[colors[label] for label in labels], label='Clustered Points')

        ## PEAKS AND TROUGHS

        ## TRAJECTORY POSITION MAGNITUDE PEAKS
        # peaks, _ = find_peaks(mag_traj, height=np.mean(mag_traj) + np.std(mag_traj))
        # troughs, _ = find_peaks(-mag_traj, height=-np.mean(mag_traj) + np.std(mag_traj))

        ## TRAJECTORY GRADIENT PEAKS
        # peaks, _ = find_peaks(np.abs(gradient_traj), threshold=np.std(gradient_traj) * 0.5)
        # troughs, _ = find_peaks(-np.abs(gradient_traj), threshold=np.std(gradient_traj) * 0.5)

        ## ORIENTATION MAGNITUDE PEAKS
        peaks, _ = find_peaks(mag_rot, height=np.mean(mag_rot) + np.std(mag_rot))
        troughs, _ = find_peaks(-mag_rot, height=-np.mean(mag_rot) + np.std(mag_rot))

        ## ORIENTATION GRADIENT PEAKS
        # peaks, _ = find_peaks(gradient_rot_mag, threshold=np.std(gradient_rot_mag) * 0.5)
        # troughs, _ = find_peaks(-gradient_rot_mag, threshold=np.std(gradient_rot_mag) * 0.5)

        ax.plot(x_traj, y_traj, z_traj, label='Setpoint Trajectory', color='dodgerblue', linewidth=2)
        ax.scatter(x_traj[peaks], y_traj[peaks], z_traj[peaks], color='red', s=50, edgecolors='black', label='Significant Points')
        ax.scatter(x_traj[troughs], y_traj[troughs], z_traj[troughs], s=50, facecolors='none', edgecolors='yellow',
                   label='Troughs')
        ax.set_title(f'Orientation Mag Trajectory Feature Tracking for {identifier}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        # plt.show()
        plt.savefig(f'Plots/features/Ori_Mag_Feature_Tracking_{identifier}.png')
        plt.close()
        print(f' Feature Tracking for {identifier} plotted successfully.')


def analyze_and_plot_trajectories(paired_datasets):
    os.makedirs('Plots/fft', exist_ok=True)
    os.makedirs('Plots/features', exist_ok=True)
    os.makedirs('Plots/features/combined', exist_ok=True)

    def magnitude(x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)

    for identifier, dataset in paired_datasets.items():
        # Extract trajectories and orientations
        x_gt, y_gt, z_gt = dataset['Position_X_setpoint'], dataset['Position_Y_setpoint'], dataset[
            'Position_Z_setpoint']
        x_filt, y_filt, z_filt = dataset['Position_X_setpoint_filtered'], dataset['Position_Y_setpoint_filtered'], \
        dataset['Position_Z_setpoint_filtered']
        x_rot, y_rot, z_rot = dataset['Orientation_X_setpoint'], dataset['Orientation_Y_setpoint'], dataset[
            'Orientation_Z_setpoint']

        # Stack for FFT and 3D plotting
        traj_gt = np.stack((x_gt, y_gt, z_gt), axis=-1)
        traj_filtered = np.stack((x_filt, y_filt, z_filt), axis=-1)
        orientation = np.stack((x_rot, y_rot, z_rot), axis=-1)

        # FFT analysis
        fft_gt = fft(traj_gt, axis=0)
        fft_filtered = fft(traj_filtered, axis=0)
        fft_difference = np.abs(fft_gt - fft_filtered)
        noise_signal = ifft(fft_difference, axis=0).real

        # Magnitude of orientation for feature tracking
        mag_rot = magnitude(x_rot, y_rot, z_rot)
        peaks, _ = find_peaks(mag_rot, height=np.mean(mag_rot) + np.std(mag_rot))
        troughs, _ = find_peaks(-mag_rot, height=-np.mean(mag_rot) + np.std(mag_rot))

        # Time vector for plotting
        time = np.linspace(0, len(x_gt), num=len(x_gt), endpoint=False)

        # Plotting FFT results
        fig_fft, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
        ax1.plot(time, traj_gt, label='Raw Data', color='gray')
        ax1.plot(time, traj_filtered, label='Filtered Data', color='green')
        ax1.set_title('Raw vs. Filtered Trajectories')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        ax2.plot(np.abs(fft_gt), label='FFT of Raw Data', color='red')
        ax2.plot(np.abs(fft_filtered), label='FFT of Filtered Data', color='blue')
        ax2.set_title('Frequency Spectra Comparison')
        ax2.set_xlabel('Frequency Index')
        ax2.set_ylabel('Magnitude')
        ax2.legend()

        ax3.plot(time, noise_signal, label='Noise Signal (Removed by Filtering)', color='purple')
        ax3.set_title('Noise Removed Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Amplitude of Noise')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f'Plots/features/combined/{identifier}_FFT_Analysis.png')
        plt.close()

        # Plotting feature tracking
        fig_features, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        ax.set_facecolor('whitesmoke')
        ax.grid(True, linestyle='--', linewidth=1, color='grey', alpha=0.5)
        ax.plot(x_gt, y_gt, z_gt, label='Setpoint Trajectory', color='dodgerblue', linewidth=2)
        ax.scatter(x_gt[peaks], y_gt[peaks], z_gt[peaks], color='red', s=50, edgecolors='black',
                   label='Significant Points')
        ax.scatter(x_gt[troughs], y_gt[troughs], z_gt[troughs], s=50, facecolors='none', edgecolors='yellow',
                   label='Troughs')
        ax.set_title(f'Orientation Mag Trajectory Feature Tracking for {identifier}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        plt.savefig(f'Plots/features/combined/FFT_PEAK_Feature_Tracking_{identifier}.png')
        plt.close()
        print(f'Feature Tracking for {identifier} plotted successfully.')

def main():
    data_directory = 'Data'
    datasets = create_datasets(data_directory)

    # plot_rmse_over_time(datasets, filtered=False)
    # plot_rmse_comparisons(datasets, filtered=False)
    # plot_trajectories(datasets, filtered=False)

    filtered_datasets = process_and_filter_data(datasets, filter_type='rls')

    # plot_rmse_over_time(filtered_datasets, filtered=True)
    # plot_trajectories(filtered_datasets, filtered=True)
    # plot_rmse_comparisons(filtered_datasets, filtered=True)
    # track_and_plot_features(filtered_datasets, filtered=True)

    # fft_analysis_and_visualization(filtered_datasets)
    # analyze_and_plot_trajectories(filtered_datasets)

if __name__ == '__main__':
    main()