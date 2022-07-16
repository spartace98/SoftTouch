import os
import numpy as np
import pandas as pd
import sys

# def get_velocity():

def get_features_with_velocity(imu_data, current_timestep, num_of_timesteps_back):
    # imu_data[:, max(0, current_timestep-3):current_timestep, :]
    num_samples, _, on_hand_sensor_dimensions = np.shape(imu_data)
    velocities = np.zeros((num_samples, num_of_timesteps_back, on_hand_sensor_dimensions))
    # print(np.shape(imu_data))

    # calculate the velocities
    for i in range(num_of_timesteps_back):
        min_idx_1 = max(current_timestep-i, 0)
        min_idx_2 = max(current_timestep-(i+1), 0)
        velocities[:, i, :] = imu_data[:, min_idx_1, :] - imu_data[:, min_idx_2, :]

    # append the velocities
    # final_features = np.zeros((num_samples, num_of_timesteps_back*2, on_hand_sensor_dimensions))
    # min_idx = max(current_timestep-num_of_timesteps_back, 0)

    # # print(current_timestep)
    # # print(np.shape(imu_data[:, min_idx:current_timestep, :]))
    # final_features[:, :num_of_timesteps_back, :] = imu_data[:, min_idx:current_timestep, :]
    # final_features[:, num_of_timesteps_back:, :] = velocities

    final_features = np.concatenate((imu_data, velocities), axis = 1)

    # print(np.shape(imu_data))
    # print(np.shape(velocities))
    # print(np.shape(final_features))
    # sys.exit()

    return final_features

def get_features_with_vel_acc(imu_data, current_timestep, num_of_timesteps_back):
    pass