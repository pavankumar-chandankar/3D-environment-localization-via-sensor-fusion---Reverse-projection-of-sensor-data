import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_tracks_3d(track_list, fusion_data, sensor_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot fusion data (e.g., fused point cloud)
    if fusion_data is not None:
        ax.scatter(fusion_data[:, 0], fusion_data[:, 1], fusion_data[:, 2], c='b', marker='o', label='Fusion Data')

    # Plot sensor data (e.g., Kinect point cloud)
    if sensor_data is not None:
        ax.scatter(sensor_data[:, 0], sensor_data[:, 1], sensor_data[:, 2], c='g', marker='^', label='Sensor Data')

    # Plot tracks
    for track in track_list:
        ax.plot(track[:, 0], track[:, 1], track[:, 2], label=f'Track {track.id}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
